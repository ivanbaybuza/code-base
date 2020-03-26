import numpy as np


def get_acc(y_pred, y_true):
    acc_list = []
    for a, b in zip(y_pred, y_true):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)



def nd(y_pred, y_true):
    """ Normalized deviation"""
    t1 = np.sum(abs(y_pred-y_true)) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2

def SMAPE(y_pred, y_true):
    s = 0
    for a, b in zip(y_pred, y_true):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a-b) / (abs(a) + abs(b))
    return s / np.size(y_true)

def nrmse(y_pred, y_true):
    """ Normalized RMSE"""
    t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2


def rmse(y_pred, y_true):
    """ RMSE """
#     t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = np.sum((y_pred - y_true) **2) / np.size(y_true)
    return np.sqrt(t1)


def get_error(X, P):
    """
    Calculate predicton error according to Eq. 28.(MOAR)
    :param X: array, Ground truth. [..., T]
    :param P: array, Predictons. [..., T]
    :return: error
    """
    T = X.shape[-1]
    e = 0
    for t in range(T):
        e += np.linalg.norm(X[..., t] - P[..., t]) / np.linalg.norm(X[..., t])
    return e / T


def eval_forecast(y_pred, y_true):
    a = get_acc(y_pred, y_true)
    b = rmse(y_pred, y_true)
    c = nd(y_pred, y_true)
    d = nrmse(y_pred, y_true)
    e = SMAPE(y_pred, y_true)
    print('ACC:\t', a)
    print('RMSE:\t', b)
    print('ND:\t', c)
    print('NRMSE:\t', d)
    print('SMAPE:\t', e)
    return [a, b, c, d, e]
