import numpy as np
import random
import pandas as pd
import tqdm

def simplex_projection(v, b=1):

    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    z = (u > (sv - b) / range(1, len(u) + 1))
    non_neg_indices = np.argwhere(z != 0)
    if len(non_neg_indices):
        rho = non_neg_indices[-1, -1]   
    else:
        rho = 0
    theta = np.maximum(0, (sv[rho] - b) / (rho + 1))
    return np.maximum(v - theta, 0)


def olmar_onestep(epsilon, y_pred, weight_o):


    ell = max([0, epsilon - y_pred @ weight_o])
    x_bar = np.mean(y_pred)    
    denom_part = y_pred - x_bar
    denominator = np.dot(denom_part, denom_part)
    if denominator != 0:
        lmbd = ell / denominator
    else:

        lmbd = 0

    weight = weight_o + lmbd * (y_pred - x_bar)
    weight = simplex_projection(weight)
    
    return weight / sum(weight)

def OLMAR(epsilon, y_pred, weight_o):

    weight = []
    for i in range(len(y_pred)):

        weight_new = olmar_onestep(epsilon, y_pred[i], weight_o)
        weight_o = weight_new
        weight.append(weight_new)

    weight = np.array(weight)

    return(weight)

def MAR(data, window):

    T, N = data.shape
    
    y_pred = np.zeros([T-window,N])
    tmp_x = np.ones([T-window,N])
    for i in range(1, window+1):
      y_pred = y_pred + 1. / tmp_x
      tmp_x = tmp_x * (data[window-i:T-i] + 1)
    y_pred /= window

    return y_pred


#The realization of Exponential Gradient and Anticor are taken from pyolps package

def eg_next_weight(last_x, last_weight, eta):

    weight = last_weight * np.exp(eta * last_x / (last_x @ last_weight))
    return weight / sum(weight)


def EG(data, tc, opts, eta=0.05):

    data = data + 1

    n, m = data.shape
    cum_ret = 1
    cumprod_ret = np.ones(n)
    daily_ret = np.ones(n)
    day_weight = np.ones(m) / m
    day_weight_o = np.zeros(m)
    daily_portfolio = np.zeros((n, m))

    
    for t in tqdm.trange(n):
        if t >= 1:
            day_weight = eg_next_weight(data[t - 1], day_weight, eta)
        
        daily_portfolio[t, :] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

    daily_ret = daily_ret - 1

    
    return daily_ret


def anticor_expert(data, weight_o, w):

    T, N = data.shape
    weight = weight_o.copy()

    if T < 2 * w:
        return weight

    LX_1 = np.log(data[T - 2 * w: T - w])
    LX_2 = np.log(data[T - w: T])
            
    mu_1 = np.mean(LX_1, axis=0)
    mu_2 = np.mean(LX_2, axis=0)
    M_cov = np.zeros((N, N))
    M_cor = np.zeros((N, N))
    n_LX_1 = LX_1 - np.repeat(mu_1.reshape(1, -1), w, axis=0)
    n_LX_2 = LX_2 - np.repeat(mu_2.reshape(1, -1), w, axis=0)
    
    Std_1 = np.diag(n_LX_1.T @ n_LX_1) / (w - 1)
    Std_2 = np.diag(n_LX_2.T @ n_LX_2) / (w - 1)
    Std_12 = Std_1.reshape(-1, 1) @ Std_2.reshape(1, -1)
    M_cov = n_LX_1.T @ n_LX_2 / (w - 1)
    
    M_cor[Std_12 == 0] = 0
    
    M_cor[Std_12 != 0] = M_cov[Std_12 != 0] / np.sqrt(Std_12[Std_12 != 0])
    
    claim = np.zeros((N, N))
    w_mu_2 = np.repeat(mu_2.reshape(-1, 1), N, axis=1)
    w_mu_1 = np.repeat(mu_2.reshape(1, -1), N, axis=0)
    
    s_12 = (w_mu_2 >= w_mu_1) & (M_cor > 0)
    claim[s_12] += M_cor[s_12]
    
    diag_M_cor = np.diag(M_cor)
    
    cor_1 = np.repeat(-diag_M_cor.reshape(-1, 1), N, axis=1)
    cor_2 = np.repeat(-diag_M_cor.reshape(1, -1), N, axis=0)
    cor_1 = np.maximum(0, cor_1)
    cor_2 = np.maximum(0, cor_2)
    claim[s_12] += cor_1[s_12] + cor_2[s_12]
    
    transfer = np.zeros((N, N))
    sum_claim = np.repeat(np.sum(claim, axis=1).reshape(-1, 1), N, axis=1)
    s_1 = np.abs(sum_claim) > 0
    w_weight_o = np.repeat(weight_o.reshape(-1, 1), N, axis=1)
    transfer[s_1] = w_weight_o[s_1] * claim[s_1] / sum_claim[s_1]
    
    transfer_ij = transfer.T - transfer
    weight -= np.sum(transfer_ij, axis=0)
        
    return weight


def anticor_kernel(data, window, exp_ret, exp_w):

    for k in range(window - 1):
        exp_w[k] = anticor_expert(data, exp_w[k], k + 2)
    numerator = 0.
    denominator = 0.
    for k in range(window - 1):
        numerator += exp_ret[k] * exp_w[k]
        denominator += exp_ret[k]
    weight = numerator / denominator
    return weight / sum(weight), exp_w


def Anticor(data, tc, opts, window=30):

    data = data + 1
    n, m = data.shape
    cum_ret = 1
    cumprod_ret = np.ones(n)
    daily_ret = np.ones(n)
    day_weight = np.ones(m) / m
    day_weight_o = np.zeros(m)
    daily_portfolio = np.zeros((n, m))
    

    exp_ret = np.ones(window - 1)
    exp_w = np.ones((window - 1, m)) / m
    


    for t in tqdm.trange(n):
        if t >= 1:
            day_weight, exp_w = anticor_kernel(data[:t], window, exp_ret, exp_w)
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]
        
        for k in range(window - 1):
            exp_ret[k] *= data[t] @ exp_w[k]
        exp_ret = exp_ret / sum(exp_ret)

    daily_ret = daily_ret - 1
    
    return daily_ret

