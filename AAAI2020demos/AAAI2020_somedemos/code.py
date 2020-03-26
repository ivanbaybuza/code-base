
# coding: utf-8

# In[4]:


from metric import *

def evaluation(result, dataset, data, total_time, n_round):
    print(dataset)
    print(data)

    f = lambda x: 0 if x < 0 else x
    result['yhat'] = result['yhat'].apply(f)

    result_arima_fo = result[result['model'] == 'arima_with_order']
    if result_arima_fo.shape[0] > 0:
        print('ARIMA_fo:')
        eval_forecast(np.array(result_arima_fo['yhat']), np.array(result_arima_fo['true']))


    result_xgb = result[result['model'] == 'xgb']
    if result_xgb.shape[0] > 0:
        print('XGBOOST:')
        eval_forecast(np.array(result_xgb['yhat']), np.array(result_xgb['true']))


    result_prophet = result[result['model'] == 'prophet_ts']
    if result_prophet.shape[0] > 0:
        print('Prophet:')
        eval_forecast(np.array(result_prophet['yhat']), np.array(result_prophet['true']))


    result_lr = result[result['model'] == 'lr']
    if result_lr.shape[0] > 0:
        print('Linear Regression:')
        eval_forecast(np.array(result_lr['yhat']), np.array(result_lr['true']))

    result_arima_auto = result[result['model'] == 'arima_ts']
    if result_arima_auto.shape[0] > 0:
        print('ARIMA auto:')
        eval_forecast(np.array(result_arima_auto['yhat']), np.array(result_arima_auto['true']))
        
    result_lasso = result[result['model'] == 'lasso']
    if result_arima_auto.shape[0] > 0:
        print('LASSO:')
        eval_forecast(np.array(result_lasso['yhat']), np.array(result_lasso['true']))    
    
    print(total_time * 15 / n_round)


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

from multiprocessing import Pool
import time

from config import *
from exp_conf import *
from model.base import build_stacking, Stacking, StackingTS, NN_Model, build_NN_model
from utility import *

def main_reg(item, ts, train_stop):
    ts_tr = ts[:train_stop]
    ts_te = ts[train_stop:train_stop + horizon]

    ts_predict_list = []
    ts_model_list = []
    
    features_ds = np.array([ts_tr[start:start+history]
                            for start in range(0, train_stop - history - 1)])
    labels_ds = np.array([ts_tr[lid]
                          for lid in range(history, train_stop-1)])

    stacking_ts_model = build_stacking(conf_item_reg, Stacking)
    stacking_ts_model.fit(features_ds, labels_ds)

    te_features = ts_tr[-history:].copy()

    for i in range(horizon):
        predict = stacking_ts_model.predict(te_features.reshape((-1, history)))
        ts_predict_list.append(predict)
        te_features = np.concatenate((te_features[1:], predict[0]))

    ts_model_list += [c['name'] for c in conf_item_reg]

    ts_predict = np.hstack(ts_predict_list)

    # 储存预测结果
    ts_result = pd.DataFrame()
    ts_result['yhat'] = ts_predict.flatten()
    ts_result['current_period'] = train_stop - 1
    fore_period = list(range(train_stop, train_stop + horizon))
    ts_result['forecast_period'] = [period
                                    for period in fore_period
                                    for _ in range(len(ts_model_list))
                                    ]
    ts_result['item'] = item
    ts_result['model'] = ts_model_list * horizon
    true_arr = ts_te
    true_arr = true_arr if len(true_arr) == horizon else true_arr + [np.nan] * (horizon - len(true_arr))
    ts_result['true'] = [true_value
                         for true_value in true_arr
                         for _ in range(len(ts_model_list))
                         ]
    return ts_result

def run_reg(data_tsx, train_stop):
    T, N = data_ts.shape
    ts_result_list = []
    if is_multi_process:
        pool = Pool(process_num)
        for n_item in range(N):
            ts = data_ts[:, n_item]
            ts_result_list.append(pool.apply_async(main_reg, args=(n_item, ts, train_stop)))
        pool.close()
        pool.join()
        ts_result_list = [v.get() for v in ts_result_list]
    else:
        for n_item in range(N):
            ts = data_ts[:, n_item]
            ts_result_list.append(main_reg(n_item, ts, train_stop))
    return pd.concat(ts_result_list)


def main_ts(item, ts, date_list, train_stop):
    ts_tr = ts[:train_stop]
    ts_te = ts[train_stop:train_stop+horizon]

    ts_predict_list = []
    ts_model_list = []

#     date_list = pd.to_datetime(pd.bdate_range(start=start, periods=train_stop + horizon, freq=freq),unit=freq)
   
    
#     features_ds = list(range(train_stop))  # 输入instance id
    features_ds = date_list[:train_stop]
    labels_ds = ts_tr           # 输入用于训练的time series
    
    stacking_ts_model = build_stacking(conf_item_ts, StackingTS)
    stacking_ts_model.fit(features_ds, labels_ds)
    ts_predict_list.append(stacking_ts_model.predict(horizon))
    ts_model_list += [c['name'] for c in conf_item_ts]

    ts_predict = np.hstack(ts_predict_list)

    # 储存预测结果
    ts_result = pd.DataFrame()
    ts_result['yhat'] = ts_predict.flatten()
    ts_result['current_period'] = date_list[train_stop-1]
#     fore_period = list(range(train_stop, train_stop + horizon))
    fore_period = date_list[-horizon:]
    ts_result['forecast_period'] = [period
                                    for period in fore_period
                                    for _ in range(len(ts_model_list))
                                    ]
    ts_result['item'] = item
    ts_result['model'] = ts_model_list * horizon
    true_arr = ts_te
    true_arr = true_arr if len(true_arr) == horizon else true_arr + [np.nan] * (horizon - len(true_arr))
    ts_result['true'] = [true_value
                         for true_value in true_arr
                         for _ in range(len(ts_model_list))
                         ]

    return ts_result


def run_ts(data_ts, date_list, train_stop):
    # data_ts: np.array, 无特征时间序列[T, N]
    T, N = data_ts.shape
    ts_result_list = []
    if is_multi_process:
        pool = Pool(process_num)
        for n_item in range(N):
            ts = data_ts[:, n_item]
            ts_result_list.append(pool.apply_async(main_ts, args=(n_item, ts, date_list, train_stop)))
        pool.close()
        pool.join()
        ts_result_list = [v.get() for v in ts_result_list]
    else:
        for n_item in range(N):
            ts = data_ts[:, n_item]
            ts_result_list.append(main_ts(n_item, ts, date_list, train_stop))
    return pd.concat(ts_result_list)


if __name__ == '__main__':

    # Load Data
    data = load_data(data_path)
    data = data[:, :num_ts]
    
    if len(data.shape) == 3:
        data_ts = data[:, :, 0]
    else:
        data_ts = data
        
    T = data.shape[0]
    date_list = pd.to_datetime(pd.bdate_range(start=start, periods=T, freq=freq), unit=freq)
    
    test_size = int(T * 0.9)
    train_rate = [0.9]
    tr_starts = [int(T*(0.9-tr)) for tr in train_rate]
    
    result_list = []
    
    for tr_start in tr_starts:
        result = None
        total_time = 0
        n_round = 0
        for tr_stop in range(test_size, T, rate):
            
            tr_data = data_ts[tr_start:, :].copy()
            tr_date_list = date_list[tr_start:]
            
            n_round += 1

            result1 = None
            result2 = None
            if ifts:
                sta = time.time()
                result1 = run_ts(tr_data, tr_date_list, tr_stop-tr_start)
                end = time.time()
                total_time = total_time + (end - sta)
            else:
                sta = time.time()
                result2 = run_reg(tr_data, tr_stop-tr_start)
                end = time.time()
                total_time = total_time + (end - sta)

            if result is None:
                result = pd.concat([result1, result2])
            else:
                result = pd.concat([result, result1, result2])
                
        # evaluation
        result_list.append([result, dataset, tr_data.shape, total_time, n_round])
            
#     print(total_time / n_round)
#     result = pd.concat([result1, result2])
#     result = pd.concat([result1, result2])
#     result.to_csv('output/{}_{}.csv'.format(dataset, train_stop))


# In[5]:


for res in result_list:
    evaluation(res[0], res[1], res[2], res[3], res[4])
   

