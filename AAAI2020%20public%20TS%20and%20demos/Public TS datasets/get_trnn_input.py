
# coding: utf-8

# In[1]:


import pickle as pkl
import numpy as np

def load_pkl(file_path):
    with open(file_path, 'rb')as file:
        data = pkl.load(file)
    return data


# [T, N] ==>[n_example, time_lag, N]

# In[2]:


def transform(X, time_lag):
    examples = [X[sta:sta+time_lag, ...]
               for sta in range(X.shape[0] - time_lag)]
    return np.stack(examples, axis=0)

def save_npy(arr, file_path):
    with open(file_path, 'wb') as file:
        pkl.dump(arr, file)


# In[9]:


traffic_small = load_pkl('traffic_small.npy')
time_lag = 15
traffic_small_trnn = transform(traffic_small, time_lag)
print(traffic_small_trnn.shape)
save_npy(traffic_small_trnn, 'traffic_small_trnn.npy')


# In[10]:


traffic_big = load_pkl('traffic_big.npy')
time_lag = 15
traffic_big_trnn = transform(traffic_big, time_lag)
print(traffic_big_trnn.shape)
save_npy(traffic_big_trnn, 'traffic_big_trnn.npy')


# In[5]:


data = load_pkl('ushcn.npy')
time_lag = 7
data = data[:,:,6,0]
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ushcn_tmax_trnn7.npy')


# In[6]:


data = load_pkl('ushcn.npy')
time_lag = 7
data = data[:,:,6,1]
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ushcn_tmin_trnn7.npy')


# In[7]:


data = load_pkl('ushcn.npy')
time_lag = 7
data = data[:,:,6,2]
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ushcn_tavg_trnn7.npy')


# In[8]:


data = load_pkl('ushcn.npy')
time_lag = 7
data = data[:,:,6,3]
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ushcn_prcp_trnn7.npy')


# In[25]:


data = load_pkl('ele_small.npy')
time_lag = 15
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ele_small_trnn.npy')


# In[26]:


data = load_pkl('ele_big.npy')
time_lag = 15
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'ele_big_trnn.npy')


# In[4]:


data = load_pkl('D1.npy')
data = data[:, :, 0]
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'D1_trnn.npy')


# In[3]:


data = load_pkl('PC_W.npy')
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'PC_W_trnn.npy')


# In[3]:


data = load_pkl('PC_M.npy')
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'PC_M_trnn.npy')


# In[3]:


data = load_pkl('aux_smooth.npy')
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'aux_smooth_trnn.npy')


# In[5]:


data = load_pkl('aux_no_0.npy')
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'aux_no_0_trnn.npy')


# In[4]:


data = load_pkl('aux_raw.npy')
time_lag = 4
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'aux_raw_trnn.npy')


# In[3]:


data = load_pkl('smoke_resized.npy')
data = np.transpose(data, (2, 0, 1))
T = data.shape[0]
data = data.reshape((T, -1))
print(data.shape)
save_npy(data, 'smoke_flatten.npy')
time_lag = 5
trnn_data= transform(data, time_lag)
print(trnn_data.shape)
save_npy(trnn_data, 'aux_raw_trnn.npy')

