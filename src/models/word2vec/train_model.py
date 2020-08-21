from torch.utils.data import DataLoader

from src.data.make_dataset import build_dataset
from src.models.train_settings import *

training, validation = build_dataset(network_directory, flair_directory, months=1)

train_loader = DataLoader(training, **params)
validation_loader = DataLoader(validation, **params)
iter_length = len(training) / batch_size


def training_iter():
    pass
