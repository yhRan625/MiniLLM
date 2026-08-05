"""
SFT 微调脚本

SFT（Supervised Fine-Tuning）的核心：教模型遵循指令、进行对话。
- 数据：(prompt, answer) 对
- Loss：只在 answer 部分计算（prompt 部分 -100）
- 学习率：比预训练小很多（1e-5 vs 3e-4），避免破坏已学知识

SFT vs 预训练的区别：
- 预训练：所有 token 都计算 loss（学习语言本身）
- SFT：只有 assistant 部分计算 loss（学习遵循指令）

本版本针对 NVIDIA T4 使用 FP16 混合精度训练：
- 前向传播使用 FP16 autocast
- 反向传播使用 GradScaler
- RMSNorm、RoPE 等敏感计算由模型内部转为 FP32
"""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.data_loader import SFTDataset, create_dataloader
from training.optimizer import create_optimizer, create_scheduler


def train_one_step(
    model,
    batch,
    optimizer,
    scaler,
    config,
):
    """完成一次前向传播、一次反向传播，并尝试更新一次模型参数。

    SFT 与预训练的主要区别：
    1. labels 中 prompt 部分为 -100
    2. cross_entropy 使用 ignore_index=-100
    3. 只有 assistant 回答部分参与 loss

    FP16 混合精度流程：
    1. autocast 下完成前向传播和 loss 计算
    2. GradScaler 放大 loss
    3. 进行一次反向传播
    4. 恢复真实梯度
    5. 梯度裁剪
    6. 检查 Inf/NaN，并尝试更新模型参数
    7. 更新 GradScaler 的缩放倍数

    返回：
        loss_value:
            当前前向传播计算得到的 loss

        update_succeeded:
            True：模型参数成功更新
            False：梯度出现 Inf/NaN，本次参数更新被跳过
    """

    # ============================================================
    # 第一步：FP16 混合精度前向传播
    # ============================================================

    with torch.autocast(
        device_type=batch["input_ids"].device.type,
        dtype=torch.float16,
        enabled=batch["input_ids"].is_cuda,
    ):
        # input_ids:
        # (batch_size, seq_len)
        #
        # logits:
        # (batch_size, seq_len, vocab_size)
        logits = model(batch["input_ids"])

        # ========================================================
        # 第二步：右移 logits 和 labels
        # ========================================================

        # input_ids:
        # [BOS] [user问题] [EOS] [assistant回答] [EOS]
        #
        # labels:
        # [-100 ... -100]      [assistant回答] [EOS]
        #
        # logits 的位置 t 用于预测位置 t+1 的 token。
        # 最后一个 logits 没有下一个 token，因此丢弃。
        # 第一个 label 没有对应的预测位置，因此丢弃。

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()

        # ========================================================
        # 第三步：计算 SFT loss
        # ========================================================

        loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ============================================================
    # 第四步：放大 loss 并进行一次反向传播
    # ============================================================

    # GradScaler 放大 loss，从而同步放大反向传播得到的梯度，
    # 降低 FP16 小梯度下溢为 0 的风险。
    scaler.scale(loss).backward()

    # ============================================================
    # 第五步：恢复真实梯度
    # ============================================================

    # 在进行梯度裁剪前，必须先除去 GradScaler 的放大倍数。
    scaler.unscale_(optimizer)

    # ============================================================
    # 第六步：梯度裁剪
    # ============================================================

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
    )

    # ============================================================
    # 第七步：尝试更新模型参数
    # ============================================================

    # 保存参数更新前的缩放倍数。
    old_scale = scaler.get_scale()

    # scaler.step() 会检查梯度：
    #
    # 没有 Inf/NaN：
    #     执行 optimizer.step()，更新模型参数
    #
    # 出现 Inf/NaN：
    #     跳过 optimizer.step()，保护模型参数
    scaler.step(optimizer)

    # 根据本次梯度状态调整缩放倍数。
    scaler.update()

    # 如果新的 scale 小于旧的 scale，
    # 说明本次发现 Inf/NaN，模型参数更新被跳过。
    update_succeeded = scaler.get_scale() >= old_scale

    # ============================================================
    # 第八步：清空梯度
    # ============================================================

    optimizer.zero_grad(set_to_none=True)

    return loss.item(), update_succeeded


def main():
    parser = argparse.ArgumentParser(
        description="SFT 微调 MiniLLM"
    )

    parser.add_argument(
        "--pretrained-path",
        type=Path,
        required=True,
        help="预训练模型路径",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "data/minimind_dataset/lora_identity.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sft"),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/bpe.model",
    )

    args = parser.parse_args()

    # ============================================================
    # 步骤 1：加载预训练模型
    # ============================================================

    config = ModelConfig()
    model = MiniLLM(config)

    # 有 CUDA GPU 时使用 GPU，否则使用 CPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"设备: {device}")

    # 从预训练 checkpoint 加载模型权重。
    print(f"加载预训练模型: {args.pretrained_path}")

    checkpoint = torch.load(
        args.pretrained_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.train()

    print(
        f"模型参数量: "
        f"{model.count_parameters() / 1e6:.1f}M"
    )

    # ============================================================
    # 步骤 2：加载 SFT 数据
    # ============================================================

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.Load(args.tokenizer_path):
        raise RuntimeError(
            f"Tokenizer 加载失败: {args.tokenizer_path}"
        )

    # SFTDataset 负责：
    # 1. 编码 prompt 和 assistant 回答
    # 2. 构造 input_ids
    # 3. 将 prompt 对应的 labels 设为 -100
    train_dataset = SFTDataset(
        args.data_path,
        tokenizer,
        args.max_length,
        args.max_lines,
    )

    train_dataloader = create_dataloader(
        train_dataset,
        args.batch_size,
    )

    print(f"训练样本数: {len(train_dataset)}")

    # ============================================================
    # 步骤 3：创建优化器、调度器和 GradScaler
    # ============================================================

    # 计划执行的前向传播和反向传播次数：
    #
    # 每遍历一次 DataLoader：
    #     len(train_dataloader) 次前向和反向
    #
    # 总共遍历 args.epochs 次。
    total_steps = len(train_dataloader) * args.epochs

    optimizer = create_optimizer(
        model,
        args.lr,
        weight_decay=0.01,
    )

    warmup_steps = int(
        total_steps * args.warmup_ratio
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
    )

    # T4 支持 FP16，因此 CUDA 环境下启用 GradScaler。
    #
    # CPU 环境下 enabled=False：
    # 可以检查代码逻辑，但不会使用 FP16 CUDA 混合精度。
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    print(f"计划训练次数: {total_steps}")
    print(f"Warmup 次数: {warmup_steps}")
    print(
        "FP16 混合精度: "
        f"{'启用' if scaler.is_enabled() else '未启用'}"
    )

    # ============================================================
    # 步骤 4：训练循环
    # ============================================================

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # step 记录已经执行的前向和反向次数。
    step = 0

    # successful_updates 记录模型参数真正更新的次数。
    successful_updates = 0

    total_loss = 0.0
    logged_steps = 0

    # 开始训练前清空梯度。
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        print()
        print("=" * 50)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 50)

        for batch in train_dataloader:
            # 将 input_ids 和 labels 搬到相同设备。
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # 完成：
            # 一次前向传播
            # 一次反向传播
            # 尝试更新一次模型参数
            loss, update_succeeded = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
            )

            total_loss += loss
            logged_steps += 1
            step += 1

            # 只有模型参数真正更新后，
            # 学习率调度器才向前推进一次。
            if update_succeeded:
                successful_updates += 1
                scheduler.step()

            # 每隔 log_interval 次前向和反向打印日志。
            if step % args.log_interval == 0:
                avg_loss = total_loss / logged_steps
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"  step {step}/{total_steps} | "
                    f"updates: {successful_updates} | "
                    f"loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"scale: {scaler.get_scale():.1f}"
                )

                total_loss = 0.0
                logged_steps = 0

            # 每隔 save_interval 次前向和反向保存 checkpoint。
            if step % args.save_interval == 0:
                checkpoint_path = (
                    output_dir / f"ckpt_step{step}.pt"
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "successful_updates": successful_updates,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

                print(
                    f"  保存 checkpoint: "
                    f"{checkpoint_path}"
                )

    # ============================================================
    # 步骤 5：保存最终模型
    # ============================================================

    final_path = output_dir / "ckpt_final.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "successful_updates": successful_updates,
            "epoch": args.epochs,
        },
        final_path,
    )

    print()
    print(
        f"SFT 完成！最终模型保存到: "
        f"{final_path}"
    )


if __name__ == "__main__":
    main()

# #///////////////分界线////////////////////////

# """
# SFT 微调脚本

# SFT（Supervised Fine-Tuning）的核心：教模型遵循指令、进行对话。
# - 数据：(prompt, answer) 对
# - Loss：只在 answer 部分计算（prompt 部分 -100）
# - 学习率：比预训练小很多（1e-5 vs 3e-4），避免破坏已学知识

# SFT vs 预训练的区别：
# - 预训练：所有 token 都计算 loss（学习语言本身）
# - SFT：只有 assistant 部分计算 loss（学习遵循指令）
# """

# import argparse
# from pathlib import Path

# import torch
# import torch.nn.functional as F
# import sentencepiece as spm

# from model.config import ModelConfig
# from model.modeling_llm import MiniLLM
# from training.optimizer import create_optimizer, create_scheduler
# from training.data_loader import SFTDataset, create_dataloader


# def train_one_step(model, batch, optimizer, config):
#     """训练一步

#     和 pretrain.py 的 train_one_step 类似，区别在于：
#     1) ignore_index=-100（SFT 的 labels 用 -100 mask prompt 部分）
#     2) scheduler.step() 在 main loop 中调用而非函数内部

#     步骤：
#     1. 前向传播
#     2. 计算 loss（ignore_index=-100，只计算 assistant 部分）
#     3. 反向传播
#     4. 梯度裁剪
#     5. 更新参数
#     6. 清零梯度
#     """
#     # 第一步：前向传播（算预测结果）
#     logits = model(batch["input_ids"])

#     # 第二步：算 loss（右移 + 交叉熵）
#     # 为什么要右移？
#     # input_ids:  [BOS] [user问题] [EOS] [assistant回答] [EOS]
#     # labels:     [-100 ... -100]       [assistant回答] [EOS]
#     #
#     # 模型在位置 0 看到 BOS，要预测下一个 token
#     # 模型在位置 1 看到 BOS+user，要预测下一个 token
#     # ...
#     # 但最后一个位置没有"下一个 token"可以预测，所以丢弃
#     shift_logits = logits[:, :-1, :].contiguous()    # 丢弃最后一个位置
#     shift_labels = batch["labels"][:, 1:].contiguous()  # 丢弃第一个位置
#     loss = F.cross_entropy(
#         shift_logits.view(-1, config.vocab_size),
#         shift_labels.view(-1),
#         ignore_index=-100,  # prompt 部分是 -100，自动跳过
#     )

#     # 第三步：反向传播（计算每个参数梯度）
#     loss.backward()

#     # 第四步：梯度裁剪（防止梯度爆炸）
#     torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

#     # 第五步：更新参数（根据梯度更新）
#     optimizer.step()

#     # 第六步：清零梯度（准备下一次反向传播）
#     optimizer.zero_grad()

#     return loss.item()


# def main():
#     parser = argparse.ArgumentParser(description="SFT 微调 MiniLLM")
#     parser.add_argument("--pretrained-path", type=Path, required=True, help="预训练模型路径")
#     parser.add_argument("--data-path", type=Path, default=Path("data/minimind_dataset/lora_identity.jsonl"))
#     parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))
#     parser.add_argument("--lr", type=float, default=1e-5)            # SFT 学习率比预训练小 30 倍
#     parser.add_argument("--epochs", type=int, default=3)              # 训练 3 轮
#     parser.add_argument("--batch-size", type=int, default=4)
#     parser.add_argument("--max-length", type=int, default=512)
#     parser.add_argument("--warmup-ratio", type=float, default=0.03)  # warmup 比例（3%）
#     parser.add_argument("--log-interval", type=int, default=50)
#     parser.add_argument("--save-interval", type=int, default=500)
#     parser.add_argument("--max-lines", type=int, default=None)       # 限制加载行数
#     parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")  # tokenizer 路径
#     args = parser.parse_args()

#     # ============================================================
#     # 步骤 1 - 加载预训练模型
#     # ============================================================
#     config = ModelConfig()  # 创建配置对象
#     model = MiniLLM(config)  # 根据配置创建模型

#     # 选择设备：有 GPU 用 GPU，没有用 CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"设备: {device}")

#     # 从预训练 checkpoint 加载权重
#     print(f"加载预训练模型: {args.pretrained_path}")
#     ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=True)
#     model.load_state_dict(ckpt["model"])
#     model = model.to(device)  # 把模型搬到 GPU
#     print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

#     # ============================================================
#     # 步骤 2 - 加载 SFT 数据
#     # ============================================================
#     # 加载 tokenizer（把文字转成 token ids 的工具）
#     tokenizer = spm.SentencePieceProcessor()
#     tokenizer.Load(args.tokenizer_path)

#     # 创建 SFT 数据集（处理对话数据，构造 labels mask）
#     train_dataset = SFTDataset(args.data_path, tokenizer, args.max_length, args.max_lines)

#     # 创建 DataLoader（每次取 batch_size 个样本，pad 到同一长度）
#     train_dataloader = create_dataloader(train_dataset, args.batch_size)
#     print(f"训练样本数: {len(train_dataset)}")

#     # ============================================================
#     # 步骤 3 - 创建优化器（lr=1e-5，比预训练小 30 倍）
#     # ============================================================
#     # 计算总步数：每个 epoch 的 batch 数 × epoch 数
#     total_steps = len(train_dataloader) * args.epochs

#     # 创建 AdamW 优化器（参数分组：bias/norm 不做 weight_decay）
#     optimizer = create_optimizer(model, args.lr, weight_decay=0.01)

#     # 创建学习率调度器（warmup + cosine decay）
#     scheduler = create_scheduler(optimizer, int(total_steps * args.warmup_ratio), total_steps)

#     # ============================================================
#     # 步骤 4 - 训练循环（按 epoch 而不是 step）
#     # ============================================================
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     step = 0           # 当前步数
#     total_loss = 0.0   # 累计 loss

#     for epoch in range(args.epochs):           # 外层：遍历整个数据集 3 次
#         print(f"\n{'='*50}")
#         print(f"Epoch {epoch+1}/{args.epochs}")
#         print(f"{'='*50}")

#         for batch in train_dataloader:         # 内层：每次取 4 条对话
#             # 把 batch 搬到 GPU
#             batch = {k: v.to(device) for k, v in batch.items()}

#             # 训练一步：前向传播 → 算 loss → 反向传播 → 更新参数
#             loss = train_one_step(model, batch, optimizer, config)

#             total_loss += loss
#             step += 1

#             # 每隔 log_interval 步打印日志
#             if step % args.log_interval == 0:
#                 avg_loss = total_loss / args.log_interval
#                 current_lr = scheduler.get_last_lr()[0]
#                 print(f"  step {step}/{total_steps} | loss: {avg_loss:.4f} | lr: {current_lr:.2e}")
#                 total_loss = 0.0

#             # 更新学习率
#             scheduler.step()

#             # 每隔 save_interval 步保存 checkpoint
#             if step % args.save_interval == 0:
#                 ckpt_path = output_dir / f"ckpt_step{step}.pt"
#                 torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
#                 print(f"  保存 checkpoint: {ckpt_path}")

#     # ============================================================
#     # 步骤 5 - 保存最终模型
#     # ============================================================
#     final_path = output_dir / "ckpt_final.pt"
#     torch.save({"model": model.state_dict(), "step": step}, final_path)
#     print(f"\nSFT 完成！最终模型保存到: {final_path}")


# if __name__ == "__main__":
#     main()
