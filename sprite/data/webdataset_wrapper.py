"""
WebDataset 包装器 (Iterable-style)
"""

import glob
from torch.utils.data import IterableDataset
from torchvision import transforms
from typing import Optional

from ..config import SpriteConfig, get_default_config
from ..tokenizer import SpriteTokenizer
from .transforms import get_default_transform


class WebDatasetWrapper(IterableDataset):
    """
    WebDataset 包装器 (Iterable-style)
    
    支持从 .tar 文件流式加载图像
    """
    
    def __init__(
        self,
        config: SpriteConfig = None,
        tokenizer: SpriteTokenizer = None,
        webdataset_path: str = None,
        transform: Optional[transforms.Compose] = None,
        shuffle: bool = True,
        epoch_size: Optional[int] = None
    ):
        """
        Args:
            config: 配置对象
            tokenizer: 分词器
            webdataset_path: WebDataset路径 (支持glob模式, 如 "data/*.tar")
            transform: 图像变换
            shuffle: 是否打乱
            epoch_size: 每个epoch的样本数 (用于控制epoch长度)
        """
        super().__init__()
        
        # 延迟导入 webdataset
        try:
            import webdataset as wds
            self.wds = wds
        except ImportError:
            raise ImportError(
                "WebDataset is required. Install with: pip install webdataset"
            )
        
        if config is None:
            config = get_default_config()
        
        if tokenizer is None:
            tokenizer = SpriteTokenizer(config)
        
        self.config = config
        self.tokenizer = tokenizer
        self.image_size = config.image.image_size
        self.webdataset_path = webdataset_path
        self.shuffle = shuffle
        self.epoch_size = epoch_size
        
        # 默认变换
        if transform is None:
            self.transform = get_default_transform(self.image_size)
        else:
            self.transform = transform
        
        # 获取所有 tar 文件
        self.tar_files = sorted(glob.glob(webdataset_path))
        if not self.tar_files:
            raise ValueError(f"No tar files found at: {webdataset_path}")
        
        print(f"Found {len(self.tar_files)} tar shards at {webdataset_path}")
        
        # 统计样本数量 (可选)
        self._length = epoch_size
    
    def _create_pipeline(self):
        """创建 WebDataset 处理管道"""
        wds = self.wds
        
        # 创建数据集
        # shardshuffle 需要是正整数(buffer大小)或False，不能是True
        shardshuffle = 100 if self.shuffle else False
        dataset = wds.WebDataset(
            self.tar_files,
            shardshuffle=shardshuffle
        )
        
        # 样本级别打乱
        if self.shuffle:
            dataset = dataset.shuffle(1000)
        
        # 解码图像
        dataset = dataset.decode("pil")
        
        # 转换为训练样本
        dataset = dataset.map(self._process_sample)
        
        return dataset
    
    def _process_sample(self, sample: dict) -> dict:
        """处理单个样本"""
        # 获取图像 (支持多种扩展名)
        image = None
        for key in ["png", "jpg", "jpeg", "webp"]:
            if key in sample:
                image = sample[key]
                break
        
        if image is None:
            raise ValueError(f"No image found in sample. Keys: {sample.keys()}")
        
        # 转换为灰度并应用变换
        if image.mode != 'L':
            image = image.convert('L')
        
        image = self.transform(image)
        
        # 确保形状正确
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        # 编码为token
        tokens = self.tokenizer.encode(image)
        tokens = tokens.squeeze(0)
        
        # input_ids: [BOS, patch1, ..., patchN, EOS] 去掉最后一个
        # labels: [patch1, ..., patchN, EOS] 去掉第一个 (BOS)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:].clone(),
            "image": image
        }
    
    def __iter__(self):
        """迭代数据集"""
        pipeline = self._create_pipeline()
        
        count = 0
        for sample in pipeline:
            yield sample
            count += 1
            
            # 如果设置了 epoch_size，达到后停止
            if self.epoch_size is not None and count >= self.epoch_size:
                break
    
    def __len__(self) -> int:
        """返回数据集长度 (估计值)"""
        if self._length is not None:
            return self._length
        # 如果没有指定，返回一个大的估计值
        return 60000  # MNIST 默认大小
