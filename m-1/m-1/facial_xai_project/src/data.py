from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_ALIASES = {"anger": "angry", "angry": "angry", "disgusted": "disgust",
                 "fearful": "fear", "happiness": "happy", "joy": "happy",
                 "sadness": "sad", "surprised": "surprise", "calm": "neutral"}
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def canonical_class(name: str) -> str:
    name = name.lower().strip()
    return CLASS_ALIASES.get(name, name)


class FER2013(Dataset):
    """Load FER-2013 CSV or an ImageFolder root with train/ and test/ directories.

    ImageFolder validation is a deterministic, stratified 10% holdout from the
    train directory. Its test directory remains untouched.
    """

    def __init__(self, csv_path: str | Path, split: str, train: bool = False):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.paths: list[Path] | None = None
        self.pixels: list[str] | None = None
        source = Path(csv_path)
        if source.is_dir():
            self._load_imagefolder(source, split)
        else:
            frame = pd.read_csv(source)
            needed = {"emotion", "pixels", "Usage"}
            missing = needed - set(frame.columns)
            if missing:
                raise ValueError(f"FER CSV is missing columns: {sorted(missing)}")
            frame = frame[frame["Usage"].map(USAGE_TO_SPLIT) == split].reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"No rows found for split {split!r}; check Usage values")
            self.labels = frame["emotion"].astype(int).to_numpy()
            if np.any((self.labels < 0) | (self.labels >= len(CLASS_NAMES))):
                raise ValueError("Expected FER-2013 labels in [0, 6]")
            self.pixels = frame["pixels"].astype(str).tolist()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.08, contrast=0.08) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def _load_imagefolder(self, root: Path, split: str):
        from torchvision.datasets import ImageFolder

        train_dir = root / "train"
        test_dir = root / "test"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise ValueError("ImageFolder root must contain train/<class>/ and test/<class>/")
        if split == "test":
            folder = ImageFolder(test_dir)
            selected = np.arange(len(folder.samples))
        else:
            folder = ImageFolder(train_dir)
            labels = np.asarray([CLASS_TO_INDEX.get(canonical_class(folder.classes[y]), -1)
                                 for _, y in folder.samples])
            if np.any(labels < 0):
                raise ValueError(f"Unexpected class folder(s): {folder.classes}")
            indices = np.arange(len(folder.samples))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.10, random_state=42, stratify=labels
            )
            selected = train_idx if split == "train" else val_idx
        self.paths = [Path(folder.samples[i][0]) for i in selected]
        self.labels = np.asarray([
            CLASS_TO_INDEX[canonical_class(folder.classes[folder.samples[i][1]])] for i in selected
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.paths is not None:
            image = Image.open(self.paths[index]).convert("L")
        else:
            values = np.fromstring(self.pixels[index], sep=" ", dtype=np.uint8)
            if values.size != 48 * 48:
                raise ValueError(f"Expected 2304 pixel values, got {values.size} at row {index}")
            image = Image.fromarray(values.reshape(48, 48), mode="L")
        return self.transform(image), int(self.labels[index])

