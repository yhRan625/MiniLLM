"""MiniLLM 预训练脚本。

训练目标为 next-token prediction：位置 t 的 logits 预测位置 t+1 的 token。
本脚本支持：
- CUDA FP16 自动混合精度与 GradScaler
- 梯度累积
- AdamW + warmup/cosine scheduler
- 梯度裁剪
- checkpoint 保存与恢复
- 可选的 mmap 验证集
- 可选的 Weights & Biases 日志
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.data_loader import (
    PretrainDataset,
    PretrainDatasetMmap,
    create_dataloader,
)
from training.optimizer import create_optimizer, create_scheduler


def compute_loss(model, batch, config, use_amp):
    """计算 next-token prediction loss。"""
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(input_ids)

        # logits[:, t] 预测 labels[:, t + 1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=config.pad_token_id,
        )

    return loss


@torch.no_grad()
def validate(model, val_dataloader, config, device, use_amp):
    """在验证集上计算平均 loss。"""
    model.eval()
    total_loss = 0.0
    num_micro_batches = 0

    for batch in val_dataloader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        loss = compute_loss(model, batch, config, use_amp)
        total_loss += loss.item()
        num_micro_batches += 1

    model.train()

    if num_micro_batches == 0:
        return 0.0
    return total_loss / num_micro_batches


def save_checkpoint(path, model, optimizer, scheduler, scaler, step):
    """保存可继续训练的 checkpoint。"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="预训练 MiniLLM")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/minimind_dataset/pretrain_t2t_mini.jsonl"),
    )
    parser.add_argument("--tokenized-data", type=Path, default=None)
    parser.add_argument(
        "--val-tokenized-data",
        type=Path,
        default=None,
        help="可选的验证集 .npy 文件；不提供则不执行验证",
    )
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrained"))
    parser.add_argument("--resume", type=Path, default=None)

    # max_steps 指成功完成的 optimizer 更新次数，不是前向传播次数。
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--max-lines", type=int, default=None)

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum 必须大于 0")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < max_steps")
    if args.log_interval <= 0 or args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("log/save/eval interval 必须大于 0")


def main():
    args = parse_args()
    validate_args(args)

    config = ModelConfig()
    model = MiniLLM(config)
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"设备: {device} | FP16 AMP: {use_amp}")
    model = model.to(device)

    if args.wandb_project:
        if HAS_WANDB:
            wandb.init(
                project=args.wandb_project,
                config={
                    "max_steps": args.max_steps,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "max_length": args.max_length,
                    "warmup_steps": args.warmup_steps,
                    "model_params": model.count_parameters(),
                },
            )
        else:
            print("警告: 指定了 --wandb-project，但没有安装 wandb，将跳过 Wandb 日志")

    # ------------------------------------------------------------
    # 数据集
    # ------------------------------------------------------------
    if args.tokenized_data is not None:
        print(f"使用预 tokenize 数据: {args.tokenized_data}")
        train_dataset = PretrainDatasetMmap(args.tokenized_data, args.max_length)
    else:
        print("警告: 正在实时 tokenize；大数据集建议先转换为 mmap 数据")
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()
        loaded = tokenizer.Load(args.tokenizer_path)
        if not loaded:
            raise FileNotFoundError(f"无法加载 tokenizer: {args.tokenizer_path}")

        print(f"词表大小: {tokenizer.GetPieceSize()}")
        train_dataset = PretrainDataset(
            args.data_path,
            tokenizer,
            args.max_length,
            args.max_lines,
        )

    if len(train_dataset) == 0:
        raise ValueError("训练集为空")

    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    print(f"训练样本数: {len(train_dataset)}")

    val_dataloader = None
    if args.val_tokenized_data is not None:
        val_dataset = PretrainDatasetMmap(args.val_tokenized_data, args.max_length)
        if len(val_dataset) == 0:
            raise ValueError("验证集为空")
        val_dataloader = create_dataloader(val_dataset, args.batch_size)
        print(f"验证样本数: {len(val_dataset)}")

    # ------------------------------------------------------------
    # 优化器、scheduler 和 FP16 GradScaler
    # ------------------------------------------------------------
    optimizer = create_optimizer(model, args.lr, config.pretrain_weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # step 只统计真正成功的 optimizer 更新。
    step = 0
    if args.resume is not None:
        print(f"从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        step = int(checkpoint.get("step", 0))
        if step > args.max_steps:
            raise ValueError(
                f"checkpoint step={step} 已经大于 max_steps={args.max_steps}"
            )
        print(f"已恢复到 step {step}，目标为 {args.max_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # micro_step_in_update 表示当前已累计了多少次 forward/backward。
    micro_step_in_update = 0
    accumulated_loss = 0.0
    log_loss = 0.0
    log_update_count = 0
    log_start_time = time.time()
    train_start_time = time.time()

    while step < args.max_steps:
        for batch in train_dataloader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            loss = compute_loss(model, batch, config, use_amp)

            # 除以 grad_accum，使累计后的梯度近似等于大 batch 的平均梯度。
            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            accumulated_loss += loss.item()
            micro_step_in_update += 1

            # 还没累计够，只做了 forward/backward，不更新参数。
            if micro_step_in_update < args.grad_accum:
                continue

            # 梯度裁剪必须在 unscale_ 之后执行。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.max_grad_norm,
            )

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # 溢出时 step 被跳过且 scale 会下降；scale 不下降说明参数已更新。
            parameter_updated = scaler.get_scale() >= old_scale
            optimizer.zero_grad(set_to_none=True)

            update_loss = accumulated_loss / args.grad_accum
            accumulated_loss = 0.0
            micro_step_in_update = 0

            if not parameter_updated:
                print("检测到 Inf/NaN：跳过本次参数更新，scheduler 和 step 均不前进")
                continue

            scheduler.step()
            step += 1
            log_loss += update_loss
            log_update_count += 1

            if step % args.log_interval == 0:
                interval_time = time.time() - log_start_time
                total_time = time.time() - train_start_time
                average_loss = log_loss / log_update_count
                average_step_time = total_time / step
                eta_hours = average_step_time * (args.max_steps - step) / 3_600
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"step {step}/{args.max_steps} | "
                    f"loss: {average_loss:.4f} | "
                    f"lr: {current_lr:.3e} | "
                    f"最近 {log_update_count} 步: {interval_time:.1f}s | "
                    f"总耗时: {total_time / 60:.1f}min | "
                    f"预计剩余: {eta_hours:.1f}h"
                )

                if args.wandb_project and HAS_WANDB:
                    wandb.log(
                        {
                            "train/loss": average_loss,
                            "train/step_time": interval_time / log_update_count,
                            "train/learning_rate": current_lr,
                            "train/total_time_min": total_time / 60,
                            "train/step": step,
                        },
                        step=step,
                    )

                log_loss = 0.0
                log_update_count = 0
                log_start_time = time.time()

            if val_dataloader is not None and step % args.eval_interval == 0:
                val_loss = validate(model, val_dataloader, config, device, use_amp)
                print(f"validation | step {step} | loss: {val_loss:.4f}")

                if args.wandb_project and HAS_WANDB:
                    wandb.log({"val/loss": val_loss}, step=step)

            if step % args.save_interval == 0:
                checkpoint_path = output_dir / f"ckpt_step{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                )
                print(f"保存 checkpoint: {checkpoint_path}")

            if step >= args.max_steps:
                break

    final_path = output_dir / "ckpt_final.pt"
    save_checkpoint(final_path, model, optimizer, scheduler, scaler, step)

    total_time = time.time() - train_start_time
    hours = int(total_time // 3_600)
    minutes = int((total_time % 3_600) // 60)
    seconds = int(total_time % 60)
    print(
        f"训练完成！总耗时: {hours}h {minutes}m {seconds}s | "
        f"共 {step} 次参数更新 | 最终模型: {final_path}"
    )

    if args.wandb_project and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()

#///////////////分界线//////////////////////////////////

"""
预训练脚本

预训练的核心目标：让模型学习语言知识（next token prediction）
- 输入：一段 token 序列
- 输出：预测下一个 token 的概率分布
- Loss：交叉熵，labels = input_ids 右移一位

预训练 vs SFT 的区别：
- 预训练：所有 token 都计算 loss（学习语言本身）
- SFT：只有 assistant 部分计算 loss（学习遵循指令）
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.optimizer import create_optimizer, create_scheduler
from training.data_loader import PretrainDataset, PretrainDatasetMmap, create_dataloader


def train_one_step(model, batch, optimizer, scheduler, config, scaler):
    import torch.nn.functional as F

    model.train()
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    optimizer.zero_grad()

    # 第1次：FP16混合精度前向传播
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=input_ids.is_cuda,
    ):
        logits = model(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
            ignore_index=config.pad_token_id,
        )

    # 第1次：使用放大后的loss进行反向传播
    scaler.scale(loss).backward()

    # 恢复真实梯度，再进行梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
    )

    # 记录更新前的scale
    old_scale = scaler.get_scale()

    # 检查Inf/NaN；正常则更新参数，异常则跳过
    scaler.step(optimizer)
    scaler.update()

    # scale下降表示本次参数更新被跳过
    parameter_updated = scaler.get_scale() >= old_scale

    # 只有参数真正更新，学习率才向前走一次
    if parameter_updated:
        scheduler.step()

    return loss.item(), parameter_updated


def validate(model, val_dataloader, config):
    """在验证集上评估

    返回：平均 loss
    """
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=input_ids.is_cuda,
            ):
                logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1),
                ignore_index=config.pad_token_id,
            )

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """预训练主函数

    流程：加载模型 → 加载数据 → 创建优化器 → 训练循环 → 保存模型

    用法：
        # smoke test（100 步验证代码正确性）
        python scripts/2_pretrain.py --max-steps 100 --batch-size 8 --log-interval 10

        # 正式预训练（50000 步）
        python scripts/2_pretrain.py
    """
    # ============================================================
    # 命令行参数解析
    # ============================================================
    parser = argparse.ArgumentParser(description="预训练 MiniLLM")
    parser.add_argument("--data-path", type=Path, default=Path("data/minimind_dataset/pretrain_t2t_mini.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrained"))
    parser.add_argument("--max-steps", type=int, default=50000)       # 最大训练步数
    parser.add_argument("--lr", type=float, default=3e-4)             # 学习率
    parser.add_argument("--batch-size", type=int, default=8)          # 每个 batch 的样本数
    parser.add_argument("--grad-accum", type=int, default=8)          # 梯度累积步数（有效 batch = batch_size * grad_accum）
    parser.add_argument("--warmup-steps", type=int, default=2000)     # 学习率 warmup 步数
    parser.add_argument("--max-length", type=int, default=1024)       # 最大序列长度
    parser.add_argument("--log-interval", type=int, default=100)      # 每隔多少步打印日志
    parser.add_argument("--save-interval", type=int, default=1000)    # 每隔多少步保存 checkpoint
    parser.add_argument("--eval-interval", type=int, default=1000)    # 每隔多少步做验证
    parser.add_argument("--max-lines", type=int, default=None)       # 限制加载行数（避免 OOM）
    parser.add_argument("--tokenized-data", type=Path, default=None) # 预 tokenize 的 .npy 文件路径（推荐）
    parser.add_argument("--resume", type=Path, default=None)         # 从 checkpoint 恢复训练
    parser.add_argument("--wandb-project", type=str, default=None)   # Wandb 项目名（None=不启用）
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")  # tokenizer 路径
    args = parser.parse_args()

    # ============================================================
    # 步骤 1 - 初始化配置和模型
    # ============================================================
    config = ModelConfig()                   # ~38M 参数配置（hidden=512, layers=12, heads=8）
    model = MiniLLM(config)                  # 创建模型（RoPE + GQA + SwiGLU + RMSNorm）
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # ============================================================
    # Wandb 初始化
    # ============================================================
    if args.wandb_project and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            config={
                "max_steps": args.max_steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_length": args.max_length,
                "warmup_steps": args.warmup_steps,
                "model_params": model.count_parameters(),
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    model = model.to(device)                 # 把模型参数搬到 GPU（.to() 同时支持 cpu/cuda）

    # ============================================================
    # 步骤 2 - 加载数据
    # ============================================================
    if args.tokenized_data:
        # 推荐方式：从预 tokenize 的 .npy 文件加载（内存映射，不占 RAM）
        print(f"使用预 tokenize 数据: {args.tokenized_data}")
        train_dataset = PretrainDatasetMmap(args.tokenized_data, args.max_length)
    else:
        # 回退方式：实时 tokenize（需要足够内存，大数据集会 OOM）
        print("警告: 使用实时 tokenize，大数据集可能 OOM，建议先运行 scripts/tokenize_to_disk.py")
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(args.tokenizer_path)
        print(f"词表大小: {tokenizer.GetPieceSize()}")
        train_dataset = PretrainDataset(args.data_path, tokenizer, args.max_length, args.max_lines)

    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    print(f"训练样本数: {len(train_dataset)}")

    # ============================================================
    # 步骤 3 - 创建优化器和调度器
    # ============================================================
    # create_optimizer：AdamW，参数分组（bias/norm/embedding 不做 weight_decay）
# 创建优化器
    optimizer = create_optimizer(model, args.lr, config.pretrain_weight_decay)

# 创建学习率调度器
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

# 创建FP16梯度缩放器
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ============================================================
    # 步骤 4 - 从 checkpoint 恢复（如果指定）
    # ============================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    step = 0          # 当前训练步数
    if args.resume:
        print(f"从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        step = checkpoint.get("step", 0)
        print(f"已恢复到 step {step}，继续训练到 {args.max_steps}")

    total_loss = 0.0   # 累计 loss（用于计算 log_interval 内的平均 loss）
    start_time = time.time()  # 训练开始时间
    train_start_time = time.time()

    while step < args.max_steps:                # 外层循环：控制总步数
        for batch in train_dataloader:          # 内层循环：遍历一遍数据集
            # 把 batch 搬到 GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 训练一步：forward → loss → backward → 梯度裁剪 → 更新参数 → 清零梯度
            loss, parameter_updated = train_one_step(model, batch, optimizer, scheduler, config, scaler)

# 如果梯度出现Inf/NaN，本次参数没有更新
            if not parameter_updated:
                print("检测到Inf/NaN，本次参数更新已跳过，scale已降低")
                continue
            total_loss += loss                  # 累计 loss
            step += 1                           # 步数 +1

            # 每隔 log_interval 步打印日志
            if step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                total_elapsed = time.time() - train_start_time
                avg_step_time = total_elapsed / step
                eta_seconds = avg_step_time * (args.max_steps - step)
                eta_hours = eta_seconds / 3600
                current_lr = scheduler.get_last_lr()[0]
                print(f"step {step}/{args.max_steps} | loss: {avg_loss:.4f} | {elapsed:.1f}s | 总耗时: {total_elapsed/60:.1f}min | 预计剩余: {eta_hours:.1f}h")

                # Wandb 记录
                if args.wandb_project and HAS_WANDB:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/step_time": elapsed / args.log_interval,
                        "train/learning_rate": current_lr,
                        "train/total_time_min": total_elapsed / 60,
                        "train/step": step,
                    })

                total_loss = 0.0
                start_time = time.time()

            # 每隔 save_interval 步保存 checkpoint（方便中断后恢复训练）
            if step % args.save_interval == 0:
                ckpt_path = output_dir / f"ckpt_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                }, ckpt_path)
                print(f"保存 checkpoint: {ckpt_path}")

            # 达到最大步数，跳出内层 for 循环
            if step >= args.max_steps:
                break

    # ============================================================
    # 步骤 5 - 保存最终模型
    # ============================================================
    final_path = output_dir / "ckpt_final.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }, final_path)

    # 打印训练总结
    total_time = time.time() - train_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"训练完成！总耗时: {hours}h {minutes}m {seconds}s | 共 {step} 步 | 最终模型: {final_path}")

    # 结束 Wandb
    if args.wandb_project and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
