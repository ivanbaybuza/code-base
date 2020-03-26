import numpy as np
import pandas as pd
import tensorly.backend as T
import tensorly as tl
#from MOAR.functions import *
from functions import *
from tensorly.decomposition import tucker
from tensorly.base import unfold, fold
from tensorly.tenalg import multi_mode_dot, mode_dot
from scipy import linalg
from svd import *
import pickle


def svd_init(tensor, modes, ranks):
    factors = []
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=ranks[index])
        factors.append(eigenvecs.T)
    return factors

def init(dims, ranks):
    factors = []
    for index, rank in enumerate(ranks):
        U_i = np.zeros((rank, dims[index]))
        mindim = min(dims[index], rank)
        for i in range(mindim):
            U_i[i][i] = 1
        factors.append(U_i)
    return factors


def random_init(dims, ranks):
    factors = []
    for index, rank in enumerate(ranks):
        U_i = np.random.rand(rank, dims[index])
        factors.append(U_i)
    return factors


def autocorr(Y, lag=10):
    """
    计算<Y(t), Y(t-0)>, ..., <Y(t), Y(t-lag)>
    :param Y: list [tensor1, tensor2, ..., tensorT]
    :param lag: int
    :return: array(k+1)
    """
    T = len(Y)
    r = []
    for l in range(lag+1):
        product = 0.0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(Y[t] * Y[tl])
        r.append(product)
    return r


def fit_ar(Y, p=10):
    r = autocorr(Y, p)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    try:
        A = linalg.inv(R).dot(r)
    except:
        A = linalg.pinv(R).dot(r)

    return A


def predict(U, X, a):

    T = X.shape[-1]
    Y = []
    m = a.shape[0]
    for t in range(T):
        X_t = X[..., t]
        modes = list(range(len(X_t.shape)))
        Y_t = multi_mode_dot(X_t, U, modes=modes)
        Y.append(Y_t)

    U_star = []
    for u in U:
        U_star.append(linalg.pinv(u))

    X_h = []
    for t in range(m, T):
        Yt_h = None
        for k in range(m):
            Yt_h = a[k] * Y[t - m + k] if Yt_h is None else Yt_h + a[k] * Y[t - m + k]
        Xt_h = multi_mode_dot(Yt_h, U_star, modes=list(range(len(Yt_h.shape))))
        X_h.append(Xt_h)

    return X_h

def get_error(X, P):
    """
    Calculate predicton error according to Eq. 28.
    :param X: array, Ground truth. [..., T]
    :param P: array, Predictons. [..., T]
    :return: error
    """
    T = X.shape[-1]
    e = 0
    for t in range(T):
        e += linalg.norm(X[..., t] - P[..., t]) / linalg.norm(X[..., t])
    return e / T

def get_acc(A, B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    acc_list = []
    for a, b in zip(A, B):
        if max(a, b) == 0:
            acc_list.append(0)
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)


def sum_frobenius_norm(U):
    norm = 0
    for u in U:
        norm += linalg.norm(u)
    return norm


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


