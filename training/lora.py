import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """带 LoRA 适配器的线性层。

    原始线性层：
        y = x @ W.T + b

    加入 LoRA：
        y = x @ W.T + b
            + (alpha / r) * dropout(x) @ A @ B

    参数形状：
        weight: (out_features, in_features)
        A:      (in_features, r)
        B:      (r, out_features)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError(f"LoRA rank 必须大于 0，当前 r={r}")

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 保留并冻结原始权重
        self.weight = nn.Parameter(
            original_linear.weight.detach().clone(),
            requires_grad=False,
        )

        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.detach().clone(),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # A: (in_features, r)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.in_features,
                r,
                device=device,
                dtype=dtype,
            )
        )

        # B: (r, out_features)
        self.lora_B = nn.Parameter(
            torch.zeros(
                r,
                self.out_features,
                device=device,
                dtype=dtype,
            )
        )

        # A 随机初始化，B 初始化为 0。
        # 因此训练开始时 LoRA 增量为 0。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(x, self.weight, self.bias)

        lora_output = (
            self.lora_dropout(x)
            @ self.lora_A
            @ self.lora_B
        )

        return base_output + self.scaling * lora_output


def _get_parent_module(
    model: nn.Module,
    module_name: str,
) -> tuple[nn.Module, str]:
    """根据完整模块名找到父模块和当前模块名。

    例如：
        module_name = "layers.0.attn.q_proj"

    返回：
        parent = model.layers[0].attn
        child_name = "q_proj"
    """
    parts = module_name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """把指定的 nn.Linear 替换为 LoRALinear。"""

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    target_modules = set(target_modules)

    # 冻结原模型参数
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 先保存列表，再修改模型，避免遍历过程中替换模块
    modules_to_replace = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # 精确匹配最后一级模块名
        leaf_name = full_name.split(".")[-1]

        if leaf_name in target_modules:
            modules_to_replace.append((full_name, module))

    for full_name, original_linear in modules_to_replace:
        parent, child_name = _get_parent_module(model, full_name)

        lora_layer = LoRALinear(
            original_linear=original_linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        setattr(parent, child_name, lora_layer)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_ratio = (
        100.0 * trainable_params / total_params
        if total_params > 0
        else 0.0
    )

    print(
        f"已对 {len(modules_to_replace)} 个线性层应用 LoRA "
        f"(rank={r}, alpha={lora_alpha})"
    )
    print(
        f"可训练参数: {trainable_params:,} / {total_params:,} "
        f"({trainable_ratio:.4f}%)"
    )

    return model


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> nn.Module:
    """把 LoRA 增量合并进原始权重，并恢复为普通 nn.Linear。

    合并公式：
        delta_weight = (A @ B).T
        merged_weight = weight + scaling * delta_weight
    """

    lora_modules = [
        (full_name, module)
        for full_name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]

    for full_name, lora_module in lora_modules:
        parent, child_name = _get_parent_module(model, full_name)

        delta_weight = (
            lora_module.lora_A @ lora_module.lora_B
        ).T

        merged_linear = nn.Linear(
            in_features=lora_module.in_features,
            out_features=lora_module.out_features,
            bias=lora_module.bias is not None,
            device=lora_module.weight.device,
            dtype=lora_module.weight.dtype,
        )

        merged_linear.weight.copy_(
            lora_module.weight
            + lora_module.scaling * delta_weight
        )

        if lora_module.bias is not None:
            merged_linear.bias.copy_(lora_module.bias)

        # 合并后的普通线性层用于推理
        merged_linear.weight.requires_grad = False

        if merged_linear.bias is not None:
            merged_linear.bias.requires_grad = False

        setattr(parent, child_name, merged_linear)

    print(f"已合并并移除 {len(lora_modules)} 个 LoRA 层")

    return model

#///////////////////分界线///////////////////

# """
# LoRA 高效微调

# LoRA（Low-Rank Adaptation）的核心思想：
#   冻结原始权重 W，在旁边加一个低秩分解 BA：
#   h = Wx + (alpha/r) * BAx

#   - A: d×r 矩阵（随机初始化）
#   - B: r×d 矩阵（初始化为 0）
#   - r: rank（越小参数越少）
#   - alpha: 缩放因子

# LoRA vs 全参数微调：
#   - 全参：1550M 参数全部训练（Qwen1.5B）
#   - LoRA r=8：只训练 ~2M 参数（0.14%）
#   - 效果差距很小，显存节省巨大

# 本模块可以在自研模型或 Qwen2.5-1.5B 上使用。
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LoRALinear(nn.Module):
#     """LoRA 线性层

#     作为 nn.Linear 的 drop-in 替换：
#     原始：y = Wx + b
#     LoRA：y = Wx + b + (alpha/r) * B(A(x))

#     继承原始权重，冻结并添加可训练的低秩适配。
#     """

#     def __init__(
#         self,
#         original_linear: nn.Linear,
#         r: int = 8,
#         lora_alpha: int = 16,
#         lora_dropout: float = 0.05,
#     ):
#         super().__init__()
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.scaling = lora_alpha / r
#         self.in_features = original_linear.in_features
#         self.out_features = original_linear.out_features

#         # 复制原始权重（冻结）
#         self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
#         if original_linear.bias is not None:
#             self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
#         else:
#             self.bias = None

#         # LoRA 参数（可训练）
#         # A 矩阵：随机初始化
#         self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
#         # B 矩阵：初始化为 0（训练开始时 ΔW=0，不影响原始模型）
#         self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))

#         self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """前向传播

#         参数：x: (..., in_features)
#         返回：(..., out_features)

#         公式：y = Wx + b + (alpha/r) * dropout(x) @ A @ B
#         """
#         # 原始线性变换
#         base_output = F.linear(x, self.weight, self.bias)
#         # LoRA 部分：dropout(x) @ A @ B * scaling
#         lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
#         return base_output + lora_output


# def _replace_module(model, target_name, new_module):
#     """按名字替换模型中的子模块

#     比如把 model.layers.0.attn.q_proj 替换为 LoRA 版本
#     """
#     parts = target_name.split(".")
#     parent = model
#     for part in parts[:-1]:
#         parent = getattr(parent, part)
#     setattr(parent, parts[-1], new_module)


# def apply_lora_to_model(
#     model,
#     r: int = 8,
#     lora_alpha: int = 16,
#     lora_dropout: float = 0.05,
#     target_modules: list = None,
# ):
#     """对模型的指定层应用 LoRA

#     参数：
#         model: 原始模型
#         r: LoRA rank
#         lora_alpha: 缩放因子
#         lora_dropout: dropout 率
#         target_modules: 要应用 LoRA 的层名列表
#             如 ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

#     步骤：
#     1. 冻结模型所有参数
#     2. 遍历模型的所有模块
#     3. 对 target_modules 中的层，替换为 LoRA 版本
#     4. 只有 LoRA 的 A, B 参数需要训练
#     """
#     if target_modules is None:
#         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                           "gate_proj", "up_proj", "down_proj"]

#     # 1. 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False

#     # 2. 遍历模型，找到目标层，替换为 LoRA 版本
#     lora_count = 0
#     for name, module in model.named_modules():
#         # 检查是否是目标层（比如 "layers.0.attn.q_proj"）
#         if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
#             # 创建 LoRA 层（drop-in 替换）
#             lora_layer = LoRALinear(
#                 module,
#                 r=r,
#                 lora_alpha=lora_alpha,
#                 lora_dropout=lora_dropout,
#             )

#             # 替换原模块
#             _replace_module(model, name, lora_layer)
#             lora_count += 1

#     print(f"已对 {lora_count} 个层应用 LoRA（rank={r}, alpha={lora_alpha}）")

#     # 3. 打印可训练参数
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


# def merge_lora_weights(model):
#     """合并 LoRA 权重到原始权重

#     推理时使用：W' = W + (alpha/r) * BA
#     合并后推理速度不变（不需要额外计算 LoRA 部分）

#     步骤：
#     1. 遍历所有 LoRALinear 层
#     2. 计算 ΔW = (α/r) * B @ A
#     3. 加到原始权重上
#     4. 删除 LoRA 参数（可选）
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, LoRALinear):
#             # A: (in_features, r), B: (r, out_features)
#             # delta_W = (A @ B)^T: (in_features, out_features) -> T -> (out_features, in_features)
#             lora_delta = (module.lora_A @ module.lora_B).T  # (out_features, in_features)
#             module.weight.data += module.scaling * lora_delta

#     print("LoRA 权重已合并到原始权重")
