
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import logging
from multiprocessing import Pool
from prob_exp_conf import *
from config_prob import *

# from model.base import build_stacking, Stacking, StackingTS, NN_Model, build_NN_model
from utility import *
from features_api import make_features
from pre_processing import DataProcessing, extract_data

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset    
from gluonts.transform import FieldName

from utility import *
import time
from metric import *


# In[ ]:


from config import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

class WaveNetWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, train_data):
        self.model = WaveNetEstimator(
            prediction_length=self.kwargs['prediction_length'],
            freq=self.kwargs['freq'],
            cardinality=[train_data.__len__()])
        self.predictor = self.model.train(train_data)
        
            
    def predict(self, test_data):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_eval_samples=self.kwargs['num_eval_samples'],  # number of sample paths we want for evaluation
        )
        return list(forecast_it)

class DeeparWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, train_data):
        self.model = DeepAREstimator(
            prediction_length=self.kwargs['prediction_length'],
            context_length=self.kwargs['context_length'],
            freq=self.kwargs['freq'],
            num_layers=self.kwargs['num_layers'],
            num_cells=self.kwargs['num_cells'],
            cell_type=self.kwargs['cell_type'],
            use_feat_dynamic_real=self.kwargs['use_feat_dynamic_real'],
            dropout_rate=self.kwargs['dropout_rate'],
            trainer=Trainer(ctx=self.kwargs['ctx'],
#                             epochs=self.kwargs['epochs'],
#                             learning_rate=self.kwargs['learning_rate'],
#                             hybridize=False,
#                             num_batches_per_epoch=self.kwargs['num_batches_per_epoch']
                           )
            )
        self.predictor = self.model.train(train_data)
            
    def predict(self, test_data):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_eval_samples=self.kwargs['num_eval_samples'],  # number of sample paths we want for evaluation
        )
        return list(forecast_it)
    
class GPWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, train_data):
        self.model = GaussianProcessEstimator(
            prediction_length=self.kwargs['prediction_length'],
            context_length=self.kwargs['context_length'],
            freq=self.kwargs['freq'],
            cardinality=train_data.__len__(),
            trainer=Trainer(ctx=self.kwargs['ctx'],
                           )
            )
        self.predictor = self.model.train(train_data)
            
    def predict(self, test_data):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_eval_samples=self.kwargs['num_eval_samples'],  # number of sample paths we want for evaluation
        )
        return list(forecast_it)


class SimpleFFWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, train_data):
        self.model = SimpleFeedForwardEstimator(
            num_hidden_dimensions=self.kwargs['num_hidden_dimensions'],
            prediction_length=self.kwargs['prediction_length'],
            context_length=self.kwargs['context_length'],
            freq=self.kwargs['freq'],
            trainer=Trainer(ctx=self.kwargs['ctx'],
                            epochs=self.kwargs['epochs'],
                            learning_rate=self.kwargs['learning_rate'],
                            hybridize=False,
                            num_batches_per_epoch=self.kwargs['num_batches_per_epoch']
                           )
            )
        self.predictor = self.model.train(train_data)
            
    def predict(self, test_data):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_eval_samples=self.kwargs['num_eval_samples'],  # number of sample paths we want for evaluation
        )
        return list(forecast_it)

reg_models = {
    'simple_feedforward': SimpleFFWrapper,
    'deepar':DeeparWrapper,
    'gp_forecaster':GPWrapper,
    'wavenet':WaveNetWrapper,
}     

class ProbModel:
    def __init__(self, **kwargs):
        self._models = {}
    def _add_model(self, name, model):
        self._models[name] = model
    def fit(self, train_data):
        for _, model in self._models.items():
            print("\nmodel {} is training".format(_))
            model.fit(train_data)
    def predict(self, test_data):
        result_list = []
        for model_name, model in self._models.items():
            forecasts = model.predict(test_data)

            i_item = 0
            for fcast, ts_entry in zip(forecasts, test_data): #计算每个item的各个预测值
                
                for c in prediction_intervals:
                    assert 0.0 <= c <=100.0 
                
                ps = [50.0] + [50.0 + f * c / 2.0
                              for c in prediction_intervals
                              for f in [-1.0, +1.0]
                              ]
                percentiles_sorted = sorted(set(ps))
                
                ps_data = np.percentile(
                    fcast._sorted_samples,
                    q=percentiles_sorted,
                    axis=0,
                    interpolation="lower",
                )
                
                i_p50 = len(percentiles_sorted) // 2
                
                p50_data = ps_data[i_p50]
                
                # prediction_interval: [left, right]
                upper_left = ps_data[3]
                lower_left = ps_data[1]
                upper_right = ps_data[4]
                lower_right = ps_data[0]
                               
                forecast_period = 0
                labels = ts_entry['target'][-horizon:]
                for i in range(horizon):
                    forecast_period += 1
                    result_list.append([i_item, model_name, current_period, forecast_period, labels[i], p50_data[i], 
                                        upper_left[i], lower_left[i], upper_right[i], lower_right[i]])
                # [i_item,model, current_period, forecast_period,true, ywhat(p50_data), upper_left, lower_left, upper_right, lower_right]
                i_item += 1
        return result_list
        

def build_prob_model(conf_prob, ProbModel):
    prob_models = ProbModel()
    for model in conf_prob:
        name = model['name']
        params = model['param']
        print(name)
        prob_models._add_model(name=name, model=reg_models.get(name, None)(**params))
    return prob_models   


def mid_to_timestamp(period_id):
    """
    将月数据的period_id转换为Timestamp格式：
    period_id(int) ==> pd.Timestamp
    """
    year = period_id // 100
    month = period_id % 100
    return pd.Timestamp(year=year, month=month, day=1, freq='M')

def get_next_month(period_id):
    year = period_id // 100
    month = period_id % 100
    if month + 1 > 12:
        return (year + 1) * 100 + 1
    else:
        return period_id + 1

def main_prob(ts, train_stop, f, start): 
    # ts: array [T, N]
    print(start)
    # Transform Data:
    N_series = ts.shape[1]
    ts_values = ts.T # (N, T)
    
    train_data = ListDataset([{FieldName.TARGET: target,
                               FieldName.START: start}
                             for (target, start) in zip(ts_values[:, :tr_stop],
                                                        [start for _ in range(N_series)])],
                             freq=f)
    test_data = ListDataset([{FieldName.TARGET: target,
                              FieldName.START: start}
                            for (target, start) in zip(ts_values[:, :train_stop+horizon],
                                                       [start for _ in range(N_series)])],
                            freq=f)
    
    # Build and train models:
    prob_models = build_prob_model(conf_prob, ProbModel)
    prob_models.fit(train_data)
    
    # Make Predictions
    result_list = prob_models.predict(test_data)
    
    # Result Transform
    # [i_item,model, current_period, forecast_period,true, yhat(p50_data), upper_left, lower_left, upper_right, lower_right]
    result_df = pd.DataFrame(result_list)
    [left, right] = prediction_intervals
    result_df.columns = ['item', 'model', 'current_period', 'forecast_period', 'true', 'yhat', 
                           'u' + str(left), 'l' + str(left), 'u' + str(right), 'l' + str(right)]
    
#     # Predictions(<0) ==> 0
#     f = lambda x: max(x, 0)
#     for col in result_df.columns[-5:]:
#         result_df[col] = result_df[col].apply(f)
    
    return result_df

import pickle
def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == '__main__':
    # ##公共部分##
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    data = load_data(data_path)
#     data = data[:40, :]
#     data = data[:, :, 0]
    print(dataset)
    print(data.shape)
    
    T = data.shape[0]
    datelist = [str(da) for da in pd.to_datetime(pd.bdate_range(start=start, periods=T, freq=freq),unit=freq)]
    tr_starts = [0]
    
    for tr_start in tr_starts:
        evaluations = []
        sta = time.time()
        R = 0
        result = None
        for tr_stop in range(int(T * 0.9), T, 1):
            R += 1
            print(tr_stop, '/', T)
            train_data = data[tr_start:, :]
            prob_results = main_prob(train_data, tr_stop-tr_start, freq, datelist[tr_start])
#             evaluations.append(eval_forecast(np.array(prob_results['yhat']), np.array(prob_results['true'])))
            if result is None:
                result = prob_results
            else:
                result = pd.concat(result, prob_results)
        end = time.time()
        print("Time Consumption: ", (end - sta) / R)
        eval_forecast(np.array(prob_results['yhat']), np.array(prob_results['true']))
#         evaluations = np.array(evaluations)
#         print(np.mean(evaluations, axis=0))

