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
    log_dir: DirectoryPath = Field('../output/logs')
    save_model_dir: DirectoryPath = Field('../output/models')
    prediction_dir: DirectoryPath = Field('../output/predictions')
    metrics_path: FilePath = Field('../output/metrics.csv')

    # data splitting
    split_path: FilePath = Field('../data/data_split.csv')
    split_type: Literal['random', 'manual', 'hybrid'] = Field('hybrid')
    train_ratio: float = Field(0.9, ge=0, le=1)
    val_ratio: float = Field(0.05, ge=0, le=1)          
    test_ratio: float = Field(0.05, ge=0, le=1)
    # manual split param            
    val_timesteps: List[int] = Field([74])
    test_timesteps: List[int] = Field([70])

    # training parameters
    random_seed: int = Field(24)
    img_size: Tuple[int, int] = Field((768, 768))
    use_augmentation: bool = Field(True)
    batch_size: int = Field(8, ge=1)
    epochs: int = Field(100, ge=1)

    # model parameters
    num_channels: int = Field(3, gt=1, le=3)
    backbone: str = Field('resnet50')
    encoder_weights: str = Field(None)
    learning_rate: float = Field(0.001, gt=0)

    @computed_field
    @property
    def img_shape(self) -> Tuple:
        return self.img_size + (self.num_channels,)

    @computed_field
    @property
    def save_log_path(self) -> FilePath:
        return Path(self.log_dir) / Path(f'unet_{self.backbone}.csv') 

    @computed_field
    @property  
    def save_model_path(self) -> FilePath:
        return Path(self.save_model_dir) / Path(f'unet_{self.backbone}.weights.h5')

    @model_validator(mode='after')
    def validate_dirs(self) -> Self:
        for field in ['img_dir', 'mask_dir', 'log_dir', 'save_model_dir', 'prediction_dir']:
            path = getattr(self, field)
            # convert to Path object and expand
            path = Path(path).expanduser().resolve()
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            setattr(self, field, path)
        return self

    @model_validator(mode='after')
    def validate_files(self) -> Self:
        for field in ['split_path', 'metrics_path']:
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