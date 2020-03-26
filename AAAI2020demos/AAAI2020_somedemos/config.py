import os
from datetime import date

is_monthly = True   # 数据格式，True:月维度格式，False:周维度格式

# 是否使用多进程
is_multi_process = True

# 多进程进程数
process_num = 15

### 模型配置 ###
# 无特征单时序数据时间序列模型选择及配置
conf_item_ts = [
#     {
#         'name': 'arima_ts',
#         'param': {
             
#         }
#     },
    
    {
        'name': 'arima_with_order',
        'param': {
            'order': [3, 1, 1]
        }
    },
    
#     {
#         'name': 'prophet_ts',
#         'param': {
#             'freq':'1D'
#         }
#     },
#     {
#         'name': 'myarima_ts',
#         'param': {
#             "p": 2,
#             "d": 0,
#             "q": 1,
#             "m": 2
#         }
#     },
#     {
#         'name': 'exps_ts',
#         'param': {}
#     }
]

# 无特征单时序数据机器学习回归模型选择及配置
conf_item_reg = [
    {
        'name': 'xgb',
        'param': {
            'silent':1,
#             'n_estimator':50,
#             'learning_rate': 0.05, #default 0.1
#             'max_depth':6, #default 6, 3-10
        }
    },
#     # {
    #     'name': 'rf',
    #     'param': {}
    # },
#     {
#         'name': 'lr',
#         'param': {}
#     },
#     {
#         'name':'lasso',
#         'param': {}
#     }
]


########################
### 输出部分参数配置 ###
########################

# 输出文件
if not os.path.exists('output'):
    os.mkdir('output')
mutil_feature_outfile = 'output/mutilmodel_result_%s.csv' % date.today().strftime("%Y%m%d")  # 有特征多模型输出文件，预测前几期
ts_outfile = 'output/ts_result_%s.csv' % date.today().strftime("%Y%m%d")   # 输出最终预测总期数预测结果，后面 size 期单独使用时间序列预测
ts_forecast_outfile = 'output/ts_allitem_result_%s.csv' % date.today().strftime("%Y%m%d")   # 无特征单时序输出文件
