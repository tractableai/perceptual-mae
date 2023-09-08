from typing import NamedTuple
from typing import Optional
from src.models.base_model import BaseModel
from src.utils.aws import S3Client

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Datasets(NamedTuple):
    data_module: pl.LightningDataModule
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

class ExperimentData(NamedTuple):
    local_experiment_data_dir: str
    remote_experiment_data_dir: str
    s3_client: Optional[S3Client]
    model: BaseModel
    dataset_loaders: Datasets
