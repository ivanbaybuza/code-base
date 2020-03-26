import os
from datetime import date, datetime

# 选择的模型及配置
# 有特征输入预测前几期模型配置
conf_mutil = [
#     {
#         'name': 'xgb',
#         'param': {
#             "n_estimators": 50,
#             "objective": "reg:squarederror",
#         }
#     },
#     {
#         'name': 'rf',
#         'param': {"n_estimators": 5}
#     },
#     {
#         'name': 'gbdt',
#         'param': {"n_estimators": 50}
#     },
#     {
#         'name': 'lr',
#         'param': {}
#     },
    {
        'name': 'arimax',
        'param': {}
    },
#     {
#         'name': 'arima',
#         'param': {}
#     },
#     {
#         'name': 'myarimax',
#         'param': {
#             "p": 1,
#             "d": 0,
#             "q": 1,
#             "m": 1
#         }
#     },
#     {
#         'name': 'myarima',
#         'param': {
#             "p": 1,
#             "d": 0,
#             "q": 1,
#             "m": 1
#         }
#     },
#     {
#         'name': 'ada',
#         'param': {}
#     },
#     {
#         'name': 'knn',
#         'param': {}
#     },
#     {
#         'name': 'svr',
#         'param': {}
#     },
#     {
#         'name': 'ridge',
#         'param': {}
#     },
#     {
#         'name': 'lgb',
#         'param': {
#             "min_child_samples": 2,
#             "min_data_in_bin": 2,
#             "n_jobs": 1
#         }
#     },
#     {
#         'name': 'cat',
#         'param': {
#             "iterations": 100,
#             "logging_level": "Silent",
#             "allow_writing_files": False
#         }
#     },
#     {
#         'name': 'extra',
#         'param': {}
#     },
#     {
#         'name': 'lasso',
#         'param': {}
#     },
#     {
#         'name': 'elnet',
#         'param': {}
#     },
]

# 无特征单时间序列预测后几期模型配置
conf_ts = [
    {
        'name': 'arima_ts',
        'param': {}
    },
#     {
#         'name': 'myarima_ts',
#         'param': {
#             "p": 1,
#             "d": 0,
#             "q": 1,
#             "m": 1
#         }
#     },
    {
        'name': 'prophet_ts',
        'param': {}
    },
#     {
#         'name': 'exps_ts',
#         'param': {}
#     }
]

## 前处理参数
obtain_method = 'file'  # 获取数据方式 三种：file impala oracle
# obtain_method = 'impala'
# user = {
#     "host": "10.194.153.92",
#     "port": 21050
# }
data_situation = 'main_plan'  # 数据场景 三种：main_plan  raw_material  eu_data
item_column_name = 'item'  # item列名称

label_col = 'qty'  # 目标值列名
# 预测期数和每次预测的特征列，要约束预测结果的特征列放在列表最后
features_list = [
    ['order_qty1'],
    ['order_qty2'],
    ['order_qty3']
]

current_period = 201806   # 训练集最终日期，即站在该日期往后预测
lag = 1   # 要添加的多少历史订单数据当特征
testsize = len(features_list)   # 多模型预测期数
size = 12 - testsize   # 后续单时间序列模型再预测期数

is_monthly = True   # 数据格式，True:月维度格式，False:周维度格式
include_history = False   # 多模型是否拟合并输出历史数据

# 输入文件
orgin_input_file = 'input/D1_origin_data.csv'   # 没处理的输入文件

# 输出文件
if not os.path.exists('output'):
    os.mkdir('output')
pred_processing_file = 'output/pred_processing_data.csv'   # 前处理后的数据文件
mutil_feature_outfile = 'output/mutilmodel_result_%s.csv' % date.today().strftime("%Y%m%d")  # 有特征多模型输出文件，预测前几期
ts_outfile = 'output/ts_result_%s.csv' % date.today().strftime("%Y%m%d")   # 输出最终预测总期数预测结果，后面 size 期单独使用时间序列预测

## 神经网络部分配置 ##
# 神经网络类模型输出文件 -JJ
nn_outfile = 'output/nn_result_test_%s.csv' % datetime.now().strftime("%Y%m%d_%H%M")   #输出最终预测结果， 预测序列周期为 testsize个

mode = 'multi' # or: 'single'
freq = '1M'

prediction_intervals = [50, 90]
conf_prob = [
#     {
#         'name': 'simple_feedforward',
#         'param': {
#             # model
#             'freq': freq,
#             'prediction_length': testsize,
#             'context_length': 3,
#             'num_hidden_dimensions': [10],
#             # train
#             'epochs': 5,
#             'num_batches_per_epoch': 100,
#             'learning_rate': 0.001,
#             'minimum_learning_rate': 5e-05,
#             'ctx': 'cpu',
#             'weight_decay': 1e-08,
#             # predict
#             'num_eval_samples': 100,
#         }
#     },
    {
        'name': 'deepar',
        'param': {
            # model
            'freq': freq,
            'prediction_length': 1,
            'context_length': 3,
            'num_layers': 2,
            'num_cells': 20,
            'cell_type': 'lstm',
            'use_feat_dynamic_real': False, # 是否加入features
            'dropout_rate': 0.1,
            # train
            'epochs': 5,
            'num_batches_per_epoch': 10,
            'learning_rate': 0.001,
            'minimum_learning_rate': 5e-05,
            'ctx': 'cpu',
            'weight_decay': 1e-08,
            # predict
            'num_eval_samples': 100,
        }
    },
#     {
#         'name': 'gp_forecaster',
#         'param': {
#             # model
#             'freq': freq,
#             'prediction_length': testsize,
#             'context_length': 3,
#             # train
#             'ctx': 'cpu',
#             # predict
#             'num_eval_samples': 100,
#         }
#     },  
#     {
#         'name': 'wavenet',
#         'param': {
#             # model
#             'freq': freq,
#             'prediction_length': testsize,
#             # predict
#             'num_eval_samples': 100,
#         }
#     },
 
]


