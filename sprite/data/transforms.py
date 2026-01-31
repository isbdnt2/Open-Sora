"""
图像变换工具
"""

from torchvision import transforms


def get_default_transform(image_size: int) -> transforms.Compose:
    """获取默认的图像变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # 转换为 [0, 1]
    ])
