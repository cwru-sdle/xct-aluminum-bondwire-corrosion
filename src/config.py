import os
from pathlib import Path
from typing_extensions import Self
from typing import List, Literal, Tuple
from pydantic import BaseModel, DirectoryPath, FilePath, Field, model_validator, computed_field

class DataConfig(BaseModel):
    """ data download and preprocessing configuration"""

    download_dir: DirectoryPath = Field('../data/raw')
    output_dir: DirectoryPath = Field('../data/processed')

    project_id: str = Field('k27v4')                # OSF project ID
    crop_dim: Tuple[int, int] = Field((768, 768))   # post preprocessed img (height, width)
    num_workers: int = Field(None)                  # number of workers for multithreading
    
    # convert paths to Path object and expand
    @model_validator(mode='after')
    def validate_dirs(self) -> Self:
        for field in ['download_dir', 'output_dir']:
            path = getattr(self, field)
            path = Path(path).expanduser().resolve()
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            setattr(self, field, path)
        return self

class ModelConfig(BaseModel):
    """ model training and prediction configuration """

    img_dir: DirectoryPath = Field('../data/processed/images')
    mask_dir: DirectoryPath = Field('../data/processed/masks')
    output_dir: DirectoryPath = Field('../models')

    # data splitting
    split_path: FilePath = Field('../data/data_split.csv')
    split_type: Literal['random', 'manual'] = Field('manual')
    train_ratio: float = Field(0.9, ge=0, le=1)
    val_ratio: float = Field(0.05, ge=0, le=1)          
    test_ratio: float = Field(0.05, ge=0, le=1)
    # manual split param            
    val_timesteps: List[int] = Field([70])
    test_timesteps: List[int] = Field([82])

    # training parameters
    random_seed: int = Field(24)
    img_size: Tuple[int, int] = Field((768, 768))
    use_augmentation: bool = Field(True)
    batch_size: int = Field(8, ge=1)
    epochs: int = Field(1, ge=1)

    # model parameters
    num_channels: int = Field(3, gt=1, le=3)
    backbone: str = Field('resnet50')
    encoder_weights: str = Field('imagenet')
    learning_rate: float = Field(0.001, gt=0)

    @computed_field
    @property  
    def save_model_path(self) -> FilePath:
        return Path(self.output_dir) / Path(f'unet_{self.backbone}.weights.h5')

    @model_validator(mode='after')
    def validate_dirs(self) -> Self:
        for field in ['img_dir', 'mask_dir', 'output_dir']:
            path = getattr(self, field)
            # convert to Path object and expand
            path = Path(path).expanduser().resolve()
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            setattr(self, field, path)
        return self

    @model_validator(mode='after')
    def validate_files(self) -> Self:
        for field in ['split_path']:
            path = getattr(self, field)
            path = Path(path).expanduser().resolve()
            setattr(self, field, path)
        return self

    @model_validator(mode='after')
    def validate_split_ratios(self) -> Self:
        total = 0
        for field in ['train_ratio', 'val_ratio', 'test_ratio']:
            total += getattr(self, field)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Split proportions must sum to 1, got {total}')
        return self