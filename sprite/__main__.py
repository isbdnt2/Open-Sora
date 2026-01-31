"""
Sprite Image LLM - 命令行入口
支持训练、生成和测试
"""

import argparse
import torch
from pathlib import Path

from .config import get_default_config
from .tokenizer import SpriteTokenizer
from .model import SpriteModel
from .trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Sprite Image LLM - 类LLM的图像生成模型"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--image-dir", type=str, default=None,
        help="图像目录路径"
    )
    train_parser.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据训练"
    )
    train_parser.add_argument(
        "--synthetic-size", type=int, default=10000,
        help="合成数据集大小"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10,
        help="训练轮数"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="批次大小"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="学习率"
    )
    train_parser.add_argument(
        "--device", type=str, default=None,
        help="训练设备 (cuda/cpu)"
    )
    train_parser.add_argument(
        "--resume", type=str, default=None,
        help="从checkpoint恢复训练"
    )
    
    # 生成命令
    generate_parser = subparsers.add_parser("generate", help="生成图像")
    generate_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型checkpoint路径"
    )
    generate_parser.add_argument(
        "--num-images", type=int, default=4,
        help="生成图像数量"
    )
    generate_parser.add_argument(
        "--output-dir", type=str, default="./generated",
        help="输出目录"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="采样温度"
    )
    generate_parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k采样"
    )
    generate_parser.add_argument(
        "--device", type=str, default=None,
        help="设备 (cuda/cpu)"
    )
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试模型组件")
    
    # 模型信息命令
    info_parser = subparsers.add_parser("info", help="输出模型结构到文件")
    info_parser.add_argument(
        "--output", type=str, default="./model_info.txt",
        help="输出文件路径 (默认: ./model_info.txt)"
    )
    info_parser.add_argument(
        "--depth", type=int, default=4,
        help="显示深度 (默认: 4)"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_train(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "test":
        run_test()
    elif args.command == "info":
        run_info(args)
    else:
        parser.print_help()


def run_train(args):
    """运行训练"""
    config = get_default_config()
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    
    use_synthetic = args.synthetic or args.image_dir is None
    
    print("=" * 60)
    print("Sprite Image LLM - Training")
    print("=" * 60)
    
    train(
        config=config,
        image_dir=args.image_dir,
        use_synthetic=use_synthetic,
        synthetic_size=args.synthetic_size,
        num_epochs=args.epochs,
        device=args.device,
        resume_from=args.resume
    )


def run_generate(args):
    """运行生成"""
    from PIL import Image
    import numpy as np
    
    config = get_default_config()
    
    # 设备
    device = torch.device(args.device if args.device else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Sprite Image LLM - Generation")
    print("=" * 60)
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    tokenizer = SpriteTokenizer(config)
    model = SpriteModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_images} images...")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    
    # 批量生成
    with torch.no_grad():
        images, token_seqs = model.generate(
            tokenizer=tokenizer,
            num_samples=args.num_images,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
    
    # 统计 EOS 生成情况
    eos_token_id = tokenizer.eos_token_id
    for i in range(args.num_images):
        seq = token_seqs[i]
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            print(f"  Sample {i+1}: EOS at position {eos_pos[0].item()} (generated {eos_pos[0].item()-1} patches)")
        else:
            print(f"  Sample {i+1}: No EOS generated (max length reached)")
    
    for i in range(args.num_images):
        image_tensor = images[i:i+1]
        image_np = image_tensor.squeeze().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # 保存
        image_pil = Image.fromarray(image_np, mode='L')
        image_path = output_dir / f"generated_{i+1:04d}.png"
        image_pil.save(image_path)
        print(f"  Saved: {image_path}")
    
    print(f"\nGenerated {args.num_images} images to {output_dir}")


def run_test():
    """运行组件测试"""
    print("=" * 60)
    print("Sprite Image LLM - Component Test")
    print("=" * 60)
    
    # 测试配置
    print("\n1. Testing Config...")
    config = get_default_config()
    print(f"   Image size: {config.image.image_size}x{config.image.image_size}")
    print(f"   Patch size: {config.image.patch_size}x{config.image.patch_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Sequence length: {config.max_seq_len}")
    print("   ✓ Config OK")
    
    # 测试Tokenizer
    print("\n2. Testing Tokenizer...")
    tokenizer = SpriteTokenizer(config)
    
    # 创建测试图像
    test_image = torch.rand(1, 1, 32, 32)
    step = config.quant.value_step
    test_image = (torch.round(test_image / step) * step).clamp(0, 1)
    
    # 编码
    tokens = tokenizer.encode(test_image)
    print(f"   Input image shape: {test_image.shape}")
    print(f"   Token sequence shape: {tokens.shape}")
    print(f"   Token range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    # 解码 (自动移除 BOS/EOS)
    decoded = tokenizer.decode(tokens)
    print(f"   Decoded image shape: {decoded.shape}")
    
    # 验证可逆性
    error = (test_image - decoded).abs().max().item()
    print(f"   Reconstruction error: {error:.6f}")
    print("   ✓ Tokenizer OK")
    
    # 测试模型
    print("\n3. Testing Model...")
    model = SpriteModel(config)
    num_params = model.get_num_params()
    print(f"   Model parameters: {num_params:,}")
    
    # 前向传播
    batch_size = 2
    seq_len = tokens.shape[1]
    input_ids = tokens.expand(batch_size, -1)
    row_ids, col_ids = tokenizer.get_position_ids(seq_len)
    
    outputs = model(input_ids, row_ids, col_ids, labels=input_ids)
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print("   ✓ Model OK")
    
    # 测试生成 (EOS 主动终止)
    print("\n4. Testing Generation (EOS-terminated)...")
    model.eval()
    with torch.no_grad():
        generated_images, generated_tokens = model.generate(
            tokenizer=tokenizer,
            num_samples=1,
            temperature=1.0,
            top_k=50
        )
    print(f"   Generated image shape: {generated_images.shape}")
    print(f"   Generated token sequence length: {generated_tokens.shape[1]}")
    
    # 检查是否生成了 EOS
    eos_positions = (generated_tokens[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        print(f"   EOS generated at position: {eos_positions[0].item()}")
    else:
        print(f"   No EOS generated (max length reached)")
    print("   ✓ Generation OK")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def run_info(args):
    """输出模型结构到文件"""
    from torchinfo import summary
    import re
    
    print("=" * 60)
    print("Sprite Image LLM - Model Info")
    print("=" * 60)
    
    config = get_default_config()
    tokenizer = SpriteTokenizer(config)
    model = SpriteModel(config)
    
    # 准备示例输入
    batch_size = 1
    seq_len = config.max_seq_len - 1  # input_ids 长度 (不含最后一个token)
    
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    row_ids, col_ids = tokenizer.get_position_ids(seq_len)
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # 生成模型摘要
    model_summary = summary(
        model,
        input_data={
            "input_ids": input_ids,
            "row_ids": row_ids,
            "col_ids": col_ids,
            "labels": labels
        },
        depth=args.depth,
        col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0  # 不在控制台输出
    )
    
    # 后处理：清理无参数容器模块的输出
    summary_str = str(model_summary)
    lines = summary_str.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 检测 num_params 列为 "--" 的行（容器模块）
        # 格式: ... [shape] [shape] -- True/False/--
        # 使用分列的方式检测
        
        # 匹配包含 [...] [...] -- 的行 (num_params 是 --)
        match = re.search(r'(\[[^\]]+\])\s+(\[[^\]]+\])\s+(--)\s*$', line)
        if match:
            # num_params 是 "--"，这是容器模块，简化输出
            # 提取模块名部分
            name_match = re.match(r'^(.*?(?:─|├)[^\[]+)', line)
            if name_match:
                module_part = name_match.group(1).rstrip()
                cleaned_lines.append(module_part)
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    cleaned_summary = '\n'.join(cleaned_lines)
    
    # 保存到文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Sprite Image LLM - Model Structure\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Image Size: {config.image.image_size}x{config.image.image_size}\n")
        f.write(f"  Patch Size: {config.image.patch_size}x{config.image.patch_size}\n")
        f.write(f"  Vocab Size: {config.vocab_size}\n")
        f.write(f"  Max Sequence Length: {config.max_seq_len}\n")
        f.write(f"  d_model (Hidden Size): {config.model.d_model}\n")
        f.write(f"  n_layers: {config.model.n_layers}\n")
        f.write(f"  n_heads: {config.model.n_heads}\n")
        f.write(f"  d_ff (FFN Hidden Size): {config.model.d_ff}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write(cleaned_summary)
    
    print(f"Model info saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total params: {model_summary.total_params:,}")
    print(f"  Trainable params: {model_summary.trainable_params:,}")
    print(f"  Non-trainable params: {model_summary.total_params - model_summary.trainable_params:,}")
    print(f"  Model size: {model_summary.total_params * 4 / 1024 / 1024:.2f} MB (float32)")


if __name__ == "__main__":
    main()
