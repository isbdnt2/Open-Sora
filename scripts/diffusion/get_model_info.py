"""
获取 Open-Sora 模型结构信息的独立脚本

使用方式:
    python scripts/diffusion/get_model_info.py configs/diffusion/train/model_info.py

功能:
    - 只构建模型，不进行训练
    - 使用 torchinfo 输出完整的模型结构
    - 支持完整参数配置（8B模型）
    - 输出到文本文件便于分析
"""

import gc
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 在导入其他模块之前，强制禁用 CUDA
# 这是最可靠的方式来确保所有操作都在 CPU/meta 上进行
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 同时禁用 CUDA 相关的延迟初始化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import torch

# 强制 PyTorch 认为 CUDA 不可用
torch.cuda.is_available = lambda: False

# 禁用 torch.compile，因为它在 CPU 上可能有问题
torch._dynamo.config.suppress_errors = True
import torch._dynamo
torch._dynamo.disable()

from torchinfo import summary

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# Mock CUDA-only libraries BEFORE importing opensora
# ============================================================
from unittest.mock import MagicMock
from einops import rearrange
import importlib.util
import types


# 创建一个带有 __spec__ 的 mock 模块
def create_mock_module(name, attrs=None):
    """创建一个带有正确 __spec__ 的 mock 模块"""
    module = types.ModuleType(name)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    return module


# 创建 mock flash_attn 函数
def mock_flash_attn_func(q, k, v):
    """CPU 兼容的 flash attention 替代"""
    # 使用 PyTorch 原生的 scaled_dot_product_attention
    # 输入格式: [B, L, H, D]
    q = q.transpose(1, 2)  # [B, H, L, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2)  # [B, L, H, D]
    return x


# 创建 mock LigerRopeFunction
class MockLigerRopeFunction:
    """Mock Liger RoPE function for CPU compatibility"""
    @staticmethod
    def apply(q, k, cos, sin):
        # 简化的 RoPE 实现 - 只保持形状正确
        B, H, L, D = q.shape
        half_D = D // 2
        
        # 重塑 cos, sin 以匹配 q, k 形状
        # cos, sin: [B, L, D//2]
        cos = cos.unsqueeze(1)  # [B, 1, L, D//2]
        sin = sin.unsqueeze(1)  # [B, 1, L, D//2]
        
        # 分割成两半
        q1, q2 = q[..., :half_D], q[..., half_D:]
        k1, k2 = k[..., :half_D], k[..., half_D:]
        
        # 应用旋转
        q_out = torch.cat([
            q1 * cos - q2 * sin,
            q2 * cos + q1 * sin
        ], dim=-1)
        
        k_out = torch.cat([
            k1 * cos - k2 * sin,
            k2 * cos + k1 * sin
        ], dim=-1)
        
        return q_out, k_out


# 创建 mock LigerRMSNormFunction
class MockLigerRMSNormFunction:
    """Mock Liger RMSNorm function for CPU compatibility"""
    @staticmethod
    def apply(x, weight, eps, offset=0.0, norm_type="llama", in_place=False):
        # 标准 RMSNorm 实现
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        x_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return (x_normed * weight).to(x_dtype)


# 创建带有 __spec__ 的 mock 模块
flash_attn_module = create_mock_module('flash_attn', {
    'flash_attn_func': mock_flash_attn_func,
})

flash_attn_interface_module = create_mock_module('flash_attn_interface', {
    'flash_attn_func': lambda q, k, v: (mock_flash_attn_func(q, k, v), None),
})

# 创建 liger_kernel 模块层级
liger_kernel_module = create_mock_module('liger_kernel')
liger_kernel_ops_module = create_mock_module('liger_kernel.ops')
liger_kernel_ops_rope_module = create_mock_module('liger_kernel.ops.rope', {
    'LigerRopeFunction': MockLigerRopeFunction,
})
liger_kernel_ops_rms_norm_module = create_mock_module('liger_kernel.ops.rms_norm', {
    'LigerRMSNormFunction': MockLigerRMSNormFunction,
})

# 设置模块层级关系
liger_kernel_ops_module.rope = liger_kernel_ops_rope_module
liger_kernel_ops_module.rms_norm = liger_kernel_ops_rms_norm_module
liger_kernel_module.ops = liger_kernel_ops_module

# 注入 mock 模块到 sys.modules
sys.modules['flash_attn'] = flash_attn_module
sys.modules['flash_attn_interface'] = flash_attn_interface_module
sys.modules['liger_kernel'] = liger_kernel_module
sys.modules['liger_kernel.ops'] = liger_kernel_ops_module
sys.modules['liger_kernel.ops.rope'] = liger_kernel_ops_rope_module
sys.modules['liger_kernel.ops.rms_norm'] = liger_kernel_ops_rms_norm_module

# 现在可以安全导入 opensora
from opensora.registry import MODELS, build_module
from opensora.utils.config import parse_configs
from opensora.utils.misc import to_torch_dtype


def format_params(num_params):
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def main():
    # ======================================================
    # 1. 解析配置
    # ======================================================
    cfg = parse_configs()
    
    # 使用 meta device - 不分配实际内存，只记录结构
    # 这样即使是 8B 参数的模型也只需要几 MB 内存
    use_meta_device = cfg.get("use_meta_device", True)
    
    # 从配置中获取精度，默认 float32
    dtype = cfg.model.get("torch_dtype", torch.float32)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    device = "meta" if use_meta_device else "cpu"
    
    # 创建输出目录
    output_dir = cfg.get("outputs", "outputs/model_info")
    os.makedirs(output_dir, exist_ok=True)
    model_info_path = os.path.join(output_dir, "full_model_structure.txt")
    
    print("=" * 80)
    print("Open-Sora Model Structure Analyzer")
    print("=" * 80)
    print(f"Device: {device} {'(zero memory allocation!)' if use_meta_device else ''}")
    print(f"Output: {model_info_path}")
    print()
    
    # ======================================================
    # 2. 构建 Diffusion Model (核心模型)
    # ======================================================
    print("Building Diffusion Model (MMDiT)...")
    
    # 通过在配置中设置 device_map="meta"，模型将在 meta device 上创建
    # 这不会分配实际内存，只记录张量的形状和元数据
    model = build_module(cfg.model, MODELS)
    
    # 检查模型是否在 meta device 上
    first_param = next(model.parameters())
    actual_device = str(first_param.device)
    print(f"  Model created on device: {actual_device}")
    
    if "meta" in actual_device:
        print("  ✓ Using meta device (zero memory allocation)")
    else:
        print(f"  ! Warning: Model is on {actual_device}, may consume memory")
    
    # 获取模型配置
    in_channels = cfg.model.get("in_channels", 64)
    context_in_dim = cfg.model.get("context_in_dim", 4096)
    vec_in_dim = cfg.model.get("vec_in_dim", 768)
    hidden_size = cfg.model.get("hidden_size", 3072)
    num_heads = cfg.model.get("num_heads", 24)
    depth = cfg.model.get("depth", 19)
    depth_single = cfg.model.get("depth_single_blocks", 38)
    axes_dim = cfg.model.get("axes_dim", [16, 56, 56])
    head_dim = sum(axes_dim)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total Parameters: {format_params(total_params)} ({total_params:,})")
    print(f"  Trainable Parameters: {format_params(trainable_params)}")
    print()
    
    # ======================================================
    # 3. 创建 dummy 输入
    # ======================================================
    print("Creating dummy inputs for torchinfo...")
    
    batch_size = 1
    seq_len_img = 256   # 示例: 16x16 patches
    seq_len_txt = 512   # T5 max_length
    
    dummy_img = torch.randn(batch_size, seq_len_img, in_channels, dtype=dtype)
    dummy_img_ids = torch.zeros(batch_size, seq_len_img, 3, dtype=dtype)
    dummy_txt = torch.randn(batch_size, seq_len_txt, context_in_dim, dtype=dtype)
    dummy_txt_ids = torch.zeros(batch_size, seq_len_txt, 3, dtype=dtype)
    dummy_timesteps = torch.rand(batch_size, dtype=dtype)
    dummy_y_vec = torch.randn(batch_size, vec_in_dim, dtype=dtype)
    dummy_guidance = torch.ones(batch_size, dtype=dtype) * 4.0
    
    # ======================================================
    # 4. 使用 torchinfo 生成模型摘要
    # ======================================================
    print("Generating model summary with torchinfo...")
    print()
    
    with open(model_info_path, "w", encoding="utf-8") as f:
        # 写入头部信息
        f.write("=" * 120 + "\n")
        f.write("OPEN-SORA 2.0 FULL MODEL STRUCTURE\n")
        f.write("=" * 120 + "\n\n")
        
        # 写入配置信息
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  hidden_size:        {hidden_size}\n")
        f.write(f"  num_heads:          {num_heads}\n")
        f.write(f"  head_dim:           {head_dim} (sum of axes_dim)\n")
        f.write(f"  mlp_ratio:          {cfg.model.get('mlp_ratio', 4.0)}\n")
        f.write(f"  depth (double):     {depth}\n")
        f.write(f"  depth (single):     {depth_single}\n")
        f.write(f"  in_channels:        {in_channels}\n")
        f.write(f"  context_in_dim:     {context_in_dim} (T5 output)\n")
        f.write(f"  vec_in_dim:         {vec_in_dim} (CLIP output)\n")
        f.write(f"  axes_dim:           {axes_dim} [T, H, W]\n")
        f.write(f"  theta:              {cfg.model.get('theta', 10000)}\n")
        f.write("\n")
        
        f.write("PARAMETER SUMMARY:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total Parameters:     {total_params:,} ({format_params(total_params)})\n")
        f.write(f"  Trainable Parameters: {trainable_params:,} ({format_params(trainable_params)})\n")
        f.write("\n")
        
        f.write("DUMMY INPUT SHAPES:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  img:       {list(dummy_img.shape)}\n")
        f.write(f"  img_ids:   {list(dummy_img_ids.shape)}\n")
        f.write(f"  txt:       {list(dummy_txt.shape)}\n")
        f.write(f"  txt_ids:   {list(dummy_txt_ids.shape)}\n")
        f.write(f"  timesteps: {list(dummy_timesteps.shape)}\n")
        f.write(f"  y_vec:     {list(dummy_y_vec.shape)}\n")
        f.write(f"  guidance:  {list(dummy_guidance.shape)}\n")
        f.write("\n")
        
        # 生成 torchinfo 摘要
        f.write("=" * 120 + "\n")
        f.write("DIFFUSION MODEL (MMDiT) - TORCHINFO SUMMARY\n")
        f.write("=" * 120 + "\n")
        
        # 生成树状模块结构（不需要前向传播，但计算推断的 shape）
        f.write("\n")
        
        # 定义序列长度常量
        SEQ_IMG = seq_len_img  # 256
        SEQ_TXT = seq_len_txt  # 512
        SEQ_TOTAL = seq_len_img + seq_len_txt  # 768
        
        def infer_seq_len_from_path(parent_name):
            """根据模块路径推断序列长度"""
            path_lower = parent_name.lower()
            
            # 单流块：处理合并后的序列
            if "single_block" in path_lower:
                return SEQ_TOTAL
            
            # 双流块中的 img 相关
            if "img" in path_lower and "txt" not in path_lower:
                return SEQ_IMG
            
            # 双流块中的 txt 相关
            if "txt" in path_lower and "img" not in path_lower:
                return SEQ_TXT
            
            # 双流块（同时处理 img 和 txt）
            if "double_block" in path_lower:
                return SEQ_TOTAL  # 实际上分开处理，但用总长度表示
            
            # 输入投影
            if "img_in" in path_lower:
                return SEQ_IMG
            if "txt_in" in path_lower or "context" in path_lower:
                return SEQ_TXT
            
            # 时间嵌入、向量嵌入（无序列维度）
            if "time" in path_lower or "vec" in path_lower or "guidance" in path_lower:
                return 1  # 标量/向量，无序列
            
            # 输出层
            if "final" in path_lower or "last" in path_lower:
                return SEQ_IMG  # 只输出 img tokens
            
            # 默认使用总序列长度
            return SEQ_TOTAL
        
        def get_layer_shapes(module, parent_name="", hidden_size=3072, seq_len=768, batch_size=1):
            """根据模块类型和上下文推断 input/output shape"""
            class_name = module.__class__.__name__
            B = batch_size
            
            # 根据路径推断序列长度
            S = infer_seq_len_from_path(parent_name)
            
            if class_name == "Linear":
                in_f = module.in_features
                out_f = module.out_features
                
                # 特殊情况：时间/向量嵌入的 MLP（无序列维度）
                path_lower = parent_name.lower()
                if any(kw in path_lower for kw in ["time", "vec", "guidance", "mlp_embedder"]):
                    return f"[{B}, {in_f}]", f"[{B}, {out_f}]"
                
                # 调制层的 Linear（输入是向量，输出是多个调制参数）
                if "modulation" in path_lower or "mod" in path_lower:
                    if out_f > in_f * 2:  # 输出比输入大很多，说明是调制
                        num_mods = out_f // hidden_size
                        return f"[{B}, {in_f}]", f"[{B}, {out_f}] ({num_mods}x{hidden_size})"
                
                # 普通 Linear 层
                return f"[{B}, {S}, {in_f}]", f"[{B}, {S}, {out_f}]"
            
            elif class_name == "LayerNorm":
                if hasattr(module, 'normalized_shape'):
                    shape = list(module.normalized_shape)
                    dim = shape[0] if shape else hidden_size
                    return f"[{B}, {S}, {dim}]", f"[{B}, {S}, {dim}]"
            
            elif class_name in ["RMSNorm", "FusedRMSNorm"]:
                # 尝试从参数推断维度
                for p in module.parameters():
                    dim = p.shape[0]
                    return f"[{B}, {S}, {dim}]", f"[{B}, {S}, {dim}]"
                return f"[{B}, {S}, ?]", f"[{B}, {S}, ?]"
            
            elif class_name == "Conv2d":
                in_c = module.in_channels
                out_c = module.out_channels
                k = module.kernel_size
                s = module.stride
                p = module.padding
                return f"[{B}, {in_c}, H, W]", f"[{B}, {out_c}, H', W']"
            
            elif class_name == "Conv3d":
                in_c = module.in_channels
                out_c = module.out_channels
                return f"[{B}, {in_c}, T, H, W]", f"[{B}, {out_c}, T', H', W']"
            
            elif class_name == "Embedding":
                num_emb = module.num_embeddings
                emb_dim = module.embedding_dim
                return f"[{B}, {S}] (int)", f"[{B}, {S}, {emb_dim}]"
            
            elif class_name in ["SiLU", "GELU", "ReLU", "Tanh", "Sigmoid"]:
                return "(same)", "(same)"
            
            elif class_name == "Dropout":
                return "(same)", "(same)"
            
            elif class_name == "Identity":
                return "(same)", "(same)"
            
            elif class_name in ["ModuleList", "Sequential"]:
                return "--", "--"
            
            elif class_name == "Modulation":
                # Modulation: vec -> 多个调制参数
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        in_f = child.in_features
                        out_f = child.out_features
                        num_mods = out_f // hidden_size if hidden_size > 0 else 6
                        return f"[{B}, {in_f}]", f"[{B}, {num_mods}, {hidden_size}]"
                return f"[{B}, {hidden_size}]", f"[{B}, 6, {hidden_size}]"
            
            elif class_name == "QKNorm":
                # QKNorm 对 attention heads 做归一化
                head_dim = sum(cfg.model.get("axes_dim", [16, 56, 56]))
                num_heads = cfg.model.get("num_heads", 24)
                for p in module.parameters():
                    dim = p.shape[0]
                    return f"[{B}, {num_heads}, {S}, {dim}]", f"[{B}, {num_heads}, {S}, {dim}]"
                return f"[{B}, {num_heads}, {S}, {head_dim}]", f"[{B}, {num_heads}, {S}, {head_dim}]"
            
            elif class_name == "MLPEmbedder":
                # 查找内部 Linear 层
                in_f, out_f = None, None
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        if in_f is None:
                            in_f = child.in_features
                        out_f = child.out_features
                if in_f and out_f:
                    return f"[{B}, {in_f}]", f"[{B}, {out_f}]"
                return f"[{B}, ?]", f"[{B}, ?]"
            
            elif class_name == "SelfAttention":
                # 查找 qkv 投影层
                for name, child in module.named_children():
                    if 'qkv' in name.lower() and isinstance(child, torch.nn.Linear):
                        in_f = child.in_features
                        return f"[{B}, {S}, {in_f}]", f"[{B}, {S}, {in_f}]"
                return f"[{B}, {S}, {hidden_size}]", f"[{B}, {S}, {hidden_size}]"
            
            elif class_name == "DoubleStreamBlock":
                return f"img:[{B},{SEQ_IMG},{hidden_size}] txt:[{B},{SEQ_TXT},{hidden_size}]", f"img:[{B},{SEQ_IMG},{hidden_size}] txt:[{B},{SEQ_TXT},{hidden_size}]"
            
            elif class_name == "SingleStreamBlock":
                return f"[{B}, {SEQ_TOTAL}, {hidden_size}]", f"[{B}, {SEQ_TOTAL}, {hidden_size}]"
            
            elif class_name == "LastLayer":
                in_ch = cfg.model.get("in_channels", 64)
                return f"[{B}, {SEQ_IMG}, {hidden_size}]", f"[{B}, {SEQ_IMG}, {in_ch}]"
            
            elif class_name == "EmbedND":
                axes_dim = cfg.model.get("axes_dim", [16, 56, 56])
                total_dim = sum(axes_dim)
                return f"[{B}, {S}, 3]", f"[{B}, 1, {S}, {total_dim}]"
            
            return "", ""
        
        def print_module_tree(module, prefix="", name="", depth=0, max_depth=6, file=None, 
                             hidden_size=3072, seq_len=768, batch_size=1, parent_name=""):
            """递归打印模块树状结构，包含推断的 shape"""
            if depth > max_depth:
                return
            
            # 获取模块类型名
            class_name = module.__class__.__name__
            
            # 计算该模块的参数量
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            param_str = f"{num_params:,}" if num_params > 0 else "--"
            
            # 获取推断的 shape
            full_name = f"{parent_name}.{name}" if parent_name else name
            input_shape, output_shape = get_layer_shapes(module, full_name, hidden_size, seq_len, batch_size)
            
            # 构建显示名称: ClassName (var_name)
            if name:
                layer_col = f"{prefix}├─{class_name} ({name})"
            else:
                layer_col = f"{prefix}├─{class_name}"
            
            # 格式化输出
            if input_shape and output_shape:
                file.write(f"{layer_col:<60} {input_shape:<28} {output_shape:<28} {param_str:>15}\n")
            else:
                file.write(f"{layer_col:<60} {'--':<28} {'--':<28} {param_str:>15}\n")
            
            # 递归处理子模块
            children = list(module.named_children())
            for i, (child_name, child_module) in enumerate(children):
                is_last = (i == len(children) - 1)
                child_prefix = prefix + ("│   " if not is_last else "    ")
                print_module_tree(child_module, child_prefix, child_name, depth + 1, max_depth, file,
                                 hidden_size, seq_len, batch_size, full_name)
        
        # 检查是否可以运行前向传播
        first_param = next(model.parameters())
        is_meta = "meta" in str(first_param.device)
        
        if not is_meta:
            # CPU 模式：尝试运行 torchinfo
            try:
                print("  Running torchinfo with forward pass...")
                model_summary = summary(
                    model,
                    input_data={
                        "img": dummy_img,
                        "img_ids": dummy_img_ids,
                        "txt": dummy_txt,
                        "txt_ids": dummy_txt_ids,
                        "timesteps": dummy_timesteps,
                        "y_vec": dummy_y_vec,
                        "guidance": dummy_guidance,
                    },
                    depth=10,
                    verbose=0,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    row_settings=["var_names"],
                )
                f.write(str(model_summary) + "\n\n")
                print("  ✓ Torchinfo summary generated with forward pass")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                f.write(f"Error running torchinfo forward pass: {e}\n\n")
                f.write(f"Full traceback:\n{error_trace}\n\n")
                print(f"  ! Torchinfo forward pass failed: {e}")
                print(f"  Full error:\n{error_trace}")
                # 回退到静态树结构
                f.write("Falling back to static module tree...\n\n")
        
        # 始终输出静态树结构作为备份
        f.write("MODULE TREE STRUCTURE:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Layer (type)':<50} {'Input Shape':<25} {'Output Shape':<25} {'Params':>12}\n")
        f.write("-" * 120 + "\n")
        
        # 打印顶级模块
        total_seq = seq_len_img + seq_len_txt
        f.write(f"MMDiTModel{' ' * 40} [batch, {total_seq}, {hidden_size}]{' ' * 5} [batch, {seq_len_img}, {in_channels}]\n")
        
        children = list(model.named_children())
        for i, (name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            prefix = "│   " if not is_last else "    "
            print_module_tree(child, "", name, 0, 5, f, hidden_size, total_seq, 1)
        
        f.write("-" * 120 + "\n")
        f.write(f"{'Total Parameters:':<100} {total_params:>12,}\n")
        f.write("\n")
        
        print("  ✓ Module tree structure generated")
        
        # 写入详细的层级参数
        f.write("=" * 120 + "\n")
        f.write("LAYER-WISE PARAMETER DETAILS\n")
        f.write("=" * 120 + "\n")
        
        # 按模块分组统计
        module_params = {}
        for name, param in model.named_parameters():
            # 获取顶级模块名
            top_module = name.split('.')[0]
            if top_module not in module_params:
                module_params[top_module] = {"count": 0, "params": 0, "layers": []}
            module_params[top_module]["count"] += 1
            module_params[top_module]["params"] += param.numel()
            module_params[top_module]["layers"].append((name, list(param.shape), param.numel()))
        
        # 写入模块统计
        f.write("\nMODULE STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Module':<25} {'Layers':>10} {'Parameters':>20} {'Percentage':>12}\n")
        f.write("-" * 80 + "\n")
        
        for module_name, info in sorted(module_params.items(), key=lambda x: -x[1]["params"]):
            pct = 100.0 * info["params"] / total_params
            f.write(f"{module_name:<25} {info['count']:>10} {info['params']:>20,} {pct:>11.2f}%\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<25} {sum(m['count'] for m in module_params.values()):>10} {total_params:>20,} {'100.00':>11}%\n")
        f.write("\n")
        
        # 写入每层详细信息
        f.write("\nDETAILED LAYER LIST:\n")
        f.write("-" * 120 + "\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: shape={list(param.shape)}, numel={param.numel():,}, trainable={param.requires_grad}\n")
        
        # ======================================================
        # 5. 分析模型结构特点
        # ======================================================
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("MODEL ARCHITECTURE ANALYSIS\n")
        f.write("=" * 120 + "\n\n")
        
        # Double Blocks 分析
        double_block_params = module_params.get("double_blocks", {}).get("params", 0)
        single_block_params = module_params.get("single_blocks", {}).get("params", 0)
        
        f.write("TRANSFORMER BLOCKS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Double Stream Blocks: {depth} layers\n")
        f.write(f"    - Parameters: {format_params(double_block_params)} ({double_block_params:,})\n")
        f.write(f"    - Per block: {format_params(double_block_params // depth if depth > 0 else 0)}\n")
        f.write(f"  Single Stream Blocks: {depth_single} layers\n")
        f.write(f"    - Parameters: {format_params(single_block_params)} ({single_block_params:,})\n")
        f.write(f"    - Per block: {format_params(single_block_params // depth_single if depth_single > 0 else 0)}\n")
        f.write("\n")
        
        # 计算 FLOPs 估计 (简化版)
        f.write("COMPUTATIONAL COMPLEXITY (Estimated):\n")
        f.write("-" * 60 + "\n")
        # Attention: 4 * seq_len^2 * hidden_size
        # MLP: 8 * seq_len * hidden_size^2
        total_seq = seq_len_img + seq_len_txt
        attn_flops = 4 * (total_seq ** 2) * hidden_size * (depth + depth_single)
        mlp_flops = 8 * total_seq * (hidden_size ** 2) * 4 * (depth + depth_single)  # mlp_ratio=4
        total_flops = attn_flops + mlp_flops
        f.write(f"  Sequence length (img + txt): {total_seq}\n")
        f.write(f"  Estimated Attention FLOPs: {format_params(attn_flops)}\n")
        f.write(f"  Estimated MLP FLOPs: {format_params(mlp_flops)}\n")
        f.write(f"  Total estimated FLOPs: {format_params(total_flops)}\n")
        f.write("\n")
        
        # 显存估计
        f.write("MEMORY ESTIMATES (BF16):\n")
        f.write("-" * 60 + "\n")
        param_memory = total_params * 2 / (1024 ** 3)  # BF16 = 2 bytes
        optimizer_memory = total_params * 8 / (1024 ** 3)  # AdamW states
        f.write(f"  Model Parameters: {param_memory:.2f} GB\n")
        f.write(f"  Optimizer States (AdamW): {optimizer_memory:.2f} GB\n")
        f.write(f"  Gradients: {param_memory:.2f} GB\n")
        f.write(f"  Estimated Training Memory: {param_memory + optimizer_memory + param_memory:.2f} GB (excluding activations)\n")
    
    print(f"Model structure saved to: {model_info_path}")
    print()
    
    # 打印简要信息到控制台
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Double Stream Blocks: {depth}")
    print(f"Single Stream Blocks: {depth_single}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Heads: {num_heads}")
    print(f"Head Dim: {head_dim}")
    print("=" * 80)
    
    # 清理
    del model, dummy_img, dummy_txt, dummy_timesteps, dummy_y_vec, dummy_guidance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
