import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from torchvision.datasets.folder import pil_loader

from opensora.registry import DATASETS

from .read_video import read_video
from .utils import get_transforms_image, get_transforms_video, is_img, map_target_fps, read_file, temporal_random_crop

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_KEYS = ("neg", "path")
K = 10000


class Iloc:
    def __init__(self, data, sharded_folder, sharded_folders, rows_per_shard):
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, index):
        return Item(
            index,
            self.data,
            self.sharded_folder,
            self.sharded_folders,
            self.rows_per_shard,
        )


class Item:
    def __init__(self, index, data, sharded_folder, sharded_folders, rows_per_shard):
        self.index = index
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, key):
        index = self.index
        if key in self.data.columns:
            return self.data[key].iloc[index]
        else:
            shard_idx = index // self.rows_per_shard
            idx = index % self.rows_per_shard
            shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
            try:
                text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
                path = text_parquet["path"].iloc[idx]
                assert path == self.data["path"].iloc[index]
            except Exception as e:
                print(f"Error reading {shard_parquet}: {e}")
                raise
            return text_parquet[key].iloc[idx]

    def to_dict(self):
        index = self.index
        ret = {}
        ret.update(self.data.iloc[index].to_dict())
        shard_idx = index // self.rows_per_shard
        idx = index % self.rows_per_shard
        shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
        try:
            text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
            path = text_parquet["path"].iloc[idx]
            assert path == self.data["path"].iloc[index]
            ret.update(text_parquet.iloc[idx].to_dict())
        except Exception as e:
            print(f"Error reading {shard_parquet}: {e}")
            ret.update({"text": ""})
        return ret


class EfficientParquet:
    def __init__(self, df, sharded_folder):
        self.data = df
        self.total_rows = len(df)
        self.rows_per_shard = (self.total_rows + K - 1) // K
        self.sharded_folder = sharded_folder
        assert os.path.exists(sharded_folder), f"Sharded folder {sharded_folder} does not exist."
        self.sharded_folders = os.listdir(sharded_folder)
        self.sharded_folders = sorted(self.sharded_folders)

    def __len__(self):
        return self.total_rows

    @property
    def iloc(self):
        return Iloc(self.data, self.sharded_folder, self.sharded_folders, self.rows_per_shard)


@DATASETS.register_module("text")
class TextDataset(torch.utils.data.Dataset):
    """
    Dataset for text data
    """

    def __init__(
        self,
        data_path: str = None,
        tokenize_fn: callable = None,
        fps_max: int = 16,
        vmaf: bool = False,
        memory_efficient: bool = False,
        **kwargs,
    ):
        self.data_path = data_path
        self.data = read_file(data_path, memory_efficient=memory_efficient)
        self.memory_efficient = memory_efficient
        self.tokenize_fn = tokenize_fn
        self.vmaf = vmaf

        if fps_max is not None:
            self.fps_max = fps_max
        else:
            self.fps_max = 999999999

    def to_efficient(self):
        if self.memory_efficient:
            addition_data_path = self.data_path.split(".")[0]
            self._data = self.data
            self.data = EfficientParquet(self._data, addition_data_path)

    def getitem(self, index: int) -> dict:
        ret = dict()
        sample = self.data.iloc[index].to_dict()
        sample_fps = sample.get("fps", np.nan)
        new_fps, sampling_interval = map_target_fps(sample_fps, self.fps_max)
        ret.update({"sampling_interval": sampling_interval})

        if "text" in sample:
            ret["text"] = sample.pop("text")
            postfixs = []
            if new_fps != 0 and self.fps_max < 999:
                postfixs.append(f"{new_fps} FPS")
            if self.vmaf and "score_vmafmotion" in sample and not np.isnan(sample["score_vmafmotion"]):
                postfixs.append(f"{int(sample['score_vmafmotion'] + 0.5)} motion score")
            postfix = " " + ", ".join(postfixs) + "." if postfixs else ""
            ret["text"] = ret["text"] + postfix
            if self.tokenize_fn is not None:
                ret.update({k: v.squeeze(0) for k, v in self.tokenize_fn(ret["text"]).items()})

        if "ref" in sample:  # i2v & v2v reference
            ret["ref"] = sample.pop("ref")

        # name of the generated sample
        if "name" in sample:  # sample name (`dataset_idx`)
            ret["name"] = sample.pop("name")
        else:
            ret["index"] = index  # use index for name
        valid_sample = {k: v for k, v in sample.items() if k in VALID_KEYS}
        ret.update(valid_sample)
        return ret

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


@DATASETS.register_module("video_text")
class VideoTextDataset(TextDataset):
    def __init__(
        self,
        transform_name: str = None,
        bucket_class: str = "Bucket",
        rand_sample_interval: int = None,  # random sample_interval value from [1, min(rand_sample_interval, video_allowed_max)]
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_name = transform_name
        self.bucket_class = bucket_class
        self.rand_sample_interval = rand_sample_interval

    def get_image(self, index: int, height: int, width: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]
        # loading
        image = pil_loader(path)

        # transform
        transform = get_transforms_image(self.transform_name, (height, width))
        image = transform(image)

        # CHW -> CTHW
        video = image.unsqueeze(1)

        return {"video": video}

    def get_video(self, index: int, num_frames: int, height: int, width: int, sampling_interval: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]

        # loading
        vframes, vinfo = read_video(path, backend="av")

        if self.rand_sample_interval is not None:
            # randomly sample from 1 - self.rand_sample_interval
            video_allowed_max = min(len(vframes) // num_frames, self.rand_sample_interval)
            sampling_interval = random.randint(1, video_allowed_max)

        # Sampling video frames
        video = temporal_random_crop(vframes, num_frames, sampling_interval)

        video = video.clone()
        del vframes

        # transform
        transform = get_transforms_video(self.transform_name, (height, width))
        video = transform(video)  # T C H W
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video}

        return ret

    def get_image_or_video(self, index: int, num_frames: int, height: int, width: int, sampling_interval: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]

        if is_img(path):
            return self.get_image(index, height, width)
        return self.get_video(index, num_frames, height, width, sampling_interval)

    def getitem(self, index: str) -> dict:
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        ret = dict()
        ret.update(super().getitem(index))
        try:
            ret.update(self.get_image_or_video(index, num_frames, height, width, ret["sampling_interval"]))
        except Exception as e:
            path = self.data.iloc[index]["path"]
            print(f"video {path}: {e}")
            return None
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module("cached_video_text")
class CachedVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        transform_name: str = None,
        bucket_class: str = "Bucket",
        rand_sample_interval: int = None,  # random sample_interval value from [1, min(rand_sample_interval, video_allowed_max)]
        cached_video: bool = False,
        cached_text: bool = False,
        return_latents_path: bool = False,
        load_original_video: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_name = transform_name
        self.bucket_class = bucket_class
        self.rand_sample_interval = rand_sample_interval
        self.cached_video = cached_video
        self.cached_text = cached_text
        self.return_latents_path = return_latents_path
        self.load_original_video = load_original_video

    def get_latents(self, path):
        try:
            latents = torch.load(path, map_location=torch.device("cpu"))
        except Exception as e:
            print(f"Error loading latents from {path}: {e}")
            return torch.zeros_like(torch.randn(1, 1, 1, 1))
        return latents

    def get_conditioning_latents(self, index: int) -> dict:
        sample = self.data.iloc[index]
        latents_path = sample["latents_path"]
        text_t5_path = sample["text_t5_path"]
        text_clip_path = sample["text_clip_path"]
        res = dict()
        if self.cached_video:
            latents = self.get_latents(latents_path)
            res["video_latents"] = latents
        if self.cached_text:
            text_t5 = self.get_latents(text_t5_path)
            text_clip = self.get_latents(text_clip_path)
            res["text_t5"] = text_t5
            res["text_clip"] = text_clip
        if self.return_latents_path:
            res["latents_path"] = latents_path
            res["text_t5_path"] = text_t5_path
            res["text_clip_path"] = text_clip_path
        return res

    def getitem(self, index: str) -> dict:
        # a hack to pass in the (time, height, width) info from sampler
        real_index, num_frames, height, width = [int(val) for val in index.split("-")]
        ret = dict()
        if self.load_original_video:
            ret.update(super().getitem(index))
        try:
            ret.update(self.get_conditioning_latents(real_index))
        except Exception as e:
            path = self.data.iloc[real_index]["path"]
            print(f"video {path}: {e}")
            return None
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module("dummy")
class DummyDataset(VideoTextDataset):
    """
    用于测试训练流程的假数据集，不需要真实数据。
    生成随机的视频张量和文本。
    继承 VideoTextDataset 以兼容 dataloader 类型检查。
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_frames: int = 17,
        height: int = 256,
        width: int = 256,
        channels: int = 3,
        **kwargs,
    ):
        # 不调用父类 __init__，因为我们不需要真实数据文件
        # 但需要设置一些必要的属性
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.channels = channels
        self.transform_name = None
        self.bucket_class = "Bucket"
        self.rand_sample_interval = None
        self.fps_max = 24  # 添加 fps_max 属性
        self.prompts = [
            "A beautiful sunset over the ocean.",
            "A cat playing with a ball of yarn.",
            "A futuristic city with flying cars.",
            "A serene forest with a flowing river.",
            "An astronaut floating in space.",
        ]
        
        # 创建一个假的 DataFrame 来模拟真实数据集的 data 属性
        # sampler 需要用到这个属性来分桶
        self.data = pd.DataFrame({
            "path": [f"dummy_video_{i}.mp4" for i in range(num_samples)],
            "text": [self.prompts[i % len(self.prompts)] for i in range(num_samples)],
            "num_frames": [num_frames] * num_samples,
            "height": [height] * num_samples,
            "width": [width] * num_samples,
            "fps": [24] * num_samples,
        })

    def to_efficient(self):
        # 假数据集不需要这个方法，但需要实现以兼容训练代码
        pass

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 解析 bucket sampler 传入的格式 "index-num_frames-height-width"
        if isinstance(index, str) and "-" in index:
            parts = index.split("-")
            idx = int(parts[0])
            num_frames = int(parts[1])
            height = int(parts[2])
            width = int(parts[3])
        else:
            idx = int(index) if isinstance(index, str) else index
            num_frames = self.num_frames
            height = self.height
            width = self.width

        # 生成随机视频张量 (C, T, H, W)
        # 注意：channels 应为 3（RGB），VAE 会将其编码为 latent
        video = torch.randn(3, num_frames, height, width)

        # 随机选择一个 prompt
        text = self.prompts[idx % len(self.prompts)]

        return {
            "video": video,
            "text": text,
        }