import torch
import numpy as np
import random
from tqdm import tqdm
import os

def make_deterministic(seed=42):    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def indices(n_timesteps, train_part, val_part):
  
    split_ix_tr = int(n_timesteps * train_part)
    split_ix_val = int(n_timesteps * val_part)
    indices_train = list(range(split_ix_tr))
    indices_val = list(range(split_ix_tr, split_ix_val + split_ix_tr))
    indices_test = list(range(split_ix_val + split_ix_tr, n_timesteps))

    return indices_train, indices_val, indices_test


class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len)

    def __getitem__(self, index):
        return (torch.FloatTensor(self.X[index:index+self.seq_len]), torch.FloatTensor(self.y[index+self.seq_len]))


def loader(X, y, indices, window, batch_size = 1, shuffle = False):
  
    X = X[indices[0]:indices[-1]+1]
    y = y[indices[0]:indices[-1]+1]
    dataset = TimeseriesDataset(X, y, window)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers = 0)

    return loader


def Predict_Y(loader, model, scaler, device):
  
    y_true = []
    y_pred = []
  
    for batch in loader:

        X,Y = batch[:2]
        X = X.to(device)
        Y = Y.to(device)
        y_true.append(scaler.inverse_transform(Y.cpu().detach().numpy()))
        y_pred.append(scaler.inverse_transform(model(X).cpu().detach().numpy()))
    
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    return y_true, y_pred

def BollingerBands(data, tics, window, k):

    for i in tics:

        data[('BollingerBands_upper', i)] = data[('Adj Close', i)].rolling(window).mean() + k*data[('Adj Close', i)].rolling(window).std()

    for i in tics:

        data[('BollingerBands_lower', i)] = data[('Adj Close', i)].rolling(window).mean() - k*data[('Adj Close', i)].rolling(window).std()

    return data

def MACD(data, tics, window_long, window_short):

    for i in tics:

        data[('MACD', i)] = data[('Adj Close', i)].rolling(window_short).mean() - data[('Adj Close', i)].rolling(window_long).mean()

    return data


def Preprocessing(data, tics):
  
  data = data.drop(columns = 'Close')

  for name in ['Adj Close', 'High', 'Low', 'Open', 'Volume']:
    
    data[name] = data[name].pct_change()
  
  data = data.drop(index = 0)

  data = MACD(data, tics, 26, 12)
  data = BollingerBands(data, tics, 20, 2)
  data = data[26:]
  data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

  return data

