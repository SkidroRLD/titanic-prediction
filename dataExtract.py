import pandas as pd
import numpy
import torch
from pathlib import Path
from torch.utils.data import Dataset

path = Path('.')

class TravelledPassenger(Dataset):
    def __init__(self, dset) -> None:
        super(dset).__init__()
        self.labels = dset[1,:]
        self.data = dset

def read_csv(path = path):
    train = pd.read_csv(path / 'train.csv')
    test = pd.read_csv(path / 'test.csv')
    train = train.to_numpy()
    test = test.to_numpy()
    train[train == "male"] = 1
    train[train == "female"] = 0
    test[test == "male"] = 1
    test[test == "female"] = 0
    return train, test

read_csv()