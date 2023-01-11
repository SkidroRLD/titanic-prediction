import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

path = Path('.')

device = ("cuda" if torch.cuda.is_available() else "cpu")

class TravelledPassenger(Dataset):
    def __init__(self, dset) -> None:
        super().__init__()
        self.labels = dset[1,:]
        self.pid = dset[0, :]
        self.data = dset[2:,:]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return torch.asarray(self.data[index]).type(torch.float).to(device), torch.Tensor(self.pid[index]).type(torch.float).to(device), torch.Tensor(self.labels[index]).type(torch.float).to(device)

class TravellingPassenger(Dataset):
    def __init__(self, dset) -> None:
        super().__init__()
        self.pid = dset[0,:]
        self.data = dset[1:,:]
    
    def __len__(self):
        return len(self.pid)
    
    def __getitem__(self, index):
        return torch.asarray(self.data[index]).type(torch.float).to(device), torch.Tensor(self.pid[index]).type(torch.float).to(device)

def read_csv(path = path, modeltype = 'Pytorch'):
    train = pd.read_csv(path / 'train.csv')
    test = pd.read_csv(path / 'test.csv')
    train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    embarked_dict = {'S': 0, 'C': 1, 'Q': 2}
    train['Embarked'] = train['Embarked'].fillna(3)
    test['Embarked'] = test['Embarked'].fillna(3)
    train['Embarked'] = train['Embarked'].map(embarked_dict)
    test['Embarked'] = test['Embarked'].map(embarked_dict)
    if modeltype == 'Pytorch':
        train, test = convert_csv(train, test)  
    return train, test

def convert_csv(train, test):
    train = train.to_numpy()
    test = test.to_numpy()
    train[train == "male"] = 1
    train[train == "female"] = 0
    test[test == "male"] = 1
    test[test == "female"] = 0
    np.random.shuffle(train)
    train = TravelledPassenger(train) #there is a lack of ages, fix that
    test = TravellingPassenger(test)
    return train, test

read_csv()