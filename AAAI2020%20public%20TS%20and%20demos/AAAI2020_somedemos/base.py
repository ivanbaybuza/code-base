import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

from statsmodels.tsa.api import SimpleExpSmoothing
from pmdarima.arima import auto_arima, ARIMA
from fbprophet import Prophet
from .my_arima import MyArima
from .NN.NeuralNetwork import MamlWrapper, RL2Wrapper, RNNClassicWrapper, LSTMClassicWrapper


class NormalizeModel(object):
    # 归一化模型类
    def __init__(self):
        self.norm = MinMaxScaler()
        self.model = None

    def fit(self, tr_feature, tr_label, weight=None):
        tr_feature_norm = self.norm.fit_transform(tr_feature)
        if self.has_weight():
            self.model.fit(tr_feature_norm, tr_label, weight)
        else:
            self.model.fit(tr_feature_norm, tr_label)

    def predict(self, te_feature):
        te_feature_norm = self.norm.transform(te_feature)
        return self.model.predict(te_feature_norm)

    def has_weight(self):
        return True


############################################
#              Regressor                   #
############################################

class WeightRegressor():
    def __init__(self):
        self.model = None

    def fit(self, tr_feature, tr_label, weight):
        self.model.fit(tr_feature, tr_label, weight)

    def predict(self, te_feature):
        return self.model.predict(te_feature)

    def has_weight(self):
        return True


# 树模型 #
class CatBoostRegressorWrapper(object):

    def __init__(self, **kwargs):
        if "cat_features" in kwargs:
            self.cat_features = kwargs.pop("cat_features")
        else:
            self.cat_features = None
        self.logging_level = kwargs.get("cat_features")
        self.model = CatBoostRegressor(**kwargs)

    def fit(self, tr_feature, tr_label):
        train_pool = Pool(tr_feature, tr_label, cat_features=self.cat_features)
        self.model.fit(train_pool, logging_level=self.logging_level)

    def has_weight(self):
        return False

    def predict(self, te_feature):
        test_pool = Pool(te_feature, cat_features=self.cat_features)
        return self.model.predict(test_pool)


class LGBMRegressorWrapper(object):
    def __init__(self, **kwargs):
        if "cat_features" in kwargs:
            self.cat_features = kwargs.pop("cat_features")
        else:
            self.cat_features = 'auto'
        self.model = LGBMRegressor(**kwargs)

    def fit(self, tr_feature, tr_label):
        self.model.fit(tr_feature, tr_label, categorical_feature=self.cat_features)

    def has_weight(self):
        return False

    def predict(self, te_feature):
        return self.model.predict(te_feature)


class RandomForestRegressorWrapper(WeightRegressor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(**kwargs)

        
class GradientBoostingRegressorWrapper(WeightRegressor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GradientBoostingRegressor(**kwargs)
        

class XGBRegressorWrapper(WeightRegressor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = XGBRegressor(**kwargs)


class ExtraTreesRegressorWrapper(WeightRegressor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ExtraTreesRegressor(**kwargs)


# 特征需要做归一化的，非树模型 #
# 线性回归模型
class LinearRegressionWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

    @property
    def feature_importance_(self):
        return self.model.coef_


class RidgeWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = RidgeCV(**kwargs)

    @property
    def feature_importance_(self):
        return self.model.coef_


class LassoWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = LassoCV(**kwargs)

    def has_weight(self):
        return False

    @property
    def feature_importance_(self):
        return self.model.coef_


class ElasticNetWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = ElasticNetCV(**kwargs)

    def has_weight(self):
        return False

    @property
    def feature_importance_(self):
        return self.model.coef_


# SVR回归
class SVRWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVR(**kwargs)


# knn回归
class KNeighborsRegressorWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = KNeighborsRegressor(**kwargs)

    def has_weight(self):
        return False


# Adaboost回归
class AdaBoostRegressorWrapper(NormalizeModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AdaBoostRegressor(**kwargs)


# 时间序列统计学模型 #
class ArimaXWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, tr_feature, tr_label):
        self.kwargs['y'] = tr_label
        self.tr_label = tr_label
        self.kwargs['exogenous'] = tr_feature
        self.model = auto_arima(**self.kwargs)
        self.model.fit(tr_label, exogenous=tr_feature)

    def has_weight(self):
        return False

    def predict(self, te_feature):
        predict = self.model.predict(len(te_feature), exogenous=te_feature)
        if np.isnan(predict).any():
            predict = np.array([self.tr_label.mean()] * len(te_feature))
        return predict


class MyArimaXWrapper(object):

    def __init__(self, **kwargs):
        self.model = MyArima(**kwargs)

    def fit(self, tr_feature, tr_label):
        self.tr_label = tr_feature
        self.model.fit(tr_label, x=tr_feature)

    def has_weight(self):
        return False

    def predict(self, te_feature):
        predict = self.model.predict(len(te_feature), x=te_feature)
        if np.isnan(predict).any():
            predict = np.array([self.tr_label.mean()] * len(te_feature))
        return predict


# 单独时间序列预测，Arima  ExpSmoothing  Prophet  MyArima
# tr_feature：传时间索引，tr_label：传一列目标值；需要一维数组
class ProphetWrapperTS(object):

    def __init__(self, **kwargs):
        # 时间序列间隔，默认是月维度
        if 'freq' in kwargs:
            self.freq = kwargs.pop('freq')
        else:
            self.freq = 'MS'
        self.model = Prophet(**kwargs)

    def fit(self, tr_feature, tr_label):
        ds = pd.DataFrame()
        ds['ds'] = tr_feature
        ds['y'] = tr_label
        self.model.fit(ds)

    def predict(self, size):
        future = self.model.make_future_dataframe(periods=size, freq=self.freq, include_history=False)
        return self.model.predict(future)['yhat'].values


class ARIMA_with_orders(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = ARIMA(**kwargs)

    def fit(self, tr_feature, tr_label):
        self.tr_label = tr_label
        try:
            self.model.fit(y=tr_label, exogenous=None)
        except:
            self.model = None
            print("Fail to fit the model with parameters {}".format(self.kwargs))

    def predict(self, size):
        try:
            predict = self.model.predict(size)
        except:
            predict = None
        if predict is None or np.isnan(predict).any():
            predict = np.array([self.tr_label.mean()] * size)
        return predict

    
    
class ArimaWrapperTS(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, tr_feature, tr_label):
        ds = pd.Series(data=tr_label, index=tr_feature)
        self.tr_label = tr_label
        self.model = auto_arima(ds, **self.kwargs)
        self.model.fit(ds)

    def predict(self, size):
        predict = self.model.predict(size)
        if np.isnan(predict).any():
            predict = np.array([self.tr_label.mean()] * size)
        return predict


class MyArimaWrapperTS(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = MyArima(**self.kwargs)

    def fit(self, tr_feature, tr_label):
        ds = pd.Series(data=tr_label, index=tr_feature)
        self.tr_label = tr_label
        self.model.fit(ds)

    def predict(self, size):
        predict = self.model.predict(size)
        if np.isnan(predict).any():
            predict = np.array([self.tr_label.mean()] * size)
        return predict


class ExpSmoothingWrapperTS(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, tr_feature, tr_label):
        self.model = SimpleExpSmoothing(tr_label).fit()

    def predict(self, size):
        return self.model.forecast(size)


############################################
#              Stacking                    #
############################################

# 有特征值模型预测统一类(主要是sklearn预测模型)
# 也可是单独时间序列（预测后几期），以时间的年、月等当特征，拟合机器学习模型
class Stacking:
    def __init__(self):
        self.layers = []

    def add_model(self, model):
        self.layers.append(model)

    def fit(self, features, labels, weight=None):
        for model in self.layers:
            if model.has_weight():
                model.fit(features, labels, weight)
            else:
                model.fit(features, labels)

    def predict(self, features):
        next_input = []
        for model in self.layers:
            y_rf = model.predict(features).reshape(-1, 1)
            next_input.append(y_rf)
        layer_input = np.hstack(next_input)
        return layer_input


# 单独时间序列（预测后几期）模型预测统一类
class StackingTS:
    def __init__(self):
        self.layers = []

    def add_model(self, model):
        self.layers.append(model)

    def fit(self, features, labels):
        for model in self.layers:
            model.fit(features, labels)

    def predict(self, size):
        next_input = []
        for model in self.layers:
            y_rf = model.predict(size).reshape(-1, 1)
            next_input.append(y_rf)
        layer_input = np.hstack(next_input)
        return layer_input


# Available Regressison Models
reg_models = {
    "lr": LinearRegressionWrapper,
    "ridge": RidgeWrapper,
    "lasso": LassoWrapper,
    "elnet": ElasticNetWrapper,
    "knn": KNeighborsRegressorWrapper,
    "ada": AdaBoostRegressorWrapper,
    "svr": SVRWrapper,
    "xgb": XGBRegressorWrapper,
    "lgb": LGBMRegressorWrapper,
    "cat": CatBoostRegressorWrapper,
    "rf": RandomForestRegressorWrapper,
    "gbdt": GradientBoostingRegressorWrapper,
    "extra": ExtraTreesRegressorWrapper,
    "arimax": ArimaXWrapper,
    "myarimax": MyArimaXWrapper,
    ## 单时间序列模型 ##
    "arima_ts": ArimaWrapperTS,
    "prophet_ts": ProphetWrapperTS,
    "exps_ts": ExpSmoothingWrapperTS,
    "myarima_ts": MyArimaWrapperTS,
    "arima_with_order": ARIMA_with_orders,
    ## 多item神经网络模型 ##
    "maml": MamlWrapper,
    "rl2": RL2Wrapper,
    'rnn_classic': RNNClassicWrapper,
    'lstm_classic': LSTMClassicWrapper,
}


# 模型构造和启动函数
def build_stacking(conf, stack):
    """ build stacking regressor """
    stacking_model = stack()

    for model_conf in conf:
        name = model_conf["name"]
        params = model_conf["param"]
        stacking_model.add_model(model=reg_models.get(name, None)(**params))

    return stacking_model


###########################
#   神经网络模型构建       #
##########################

class NN_Model:

    def __init__(self):
        self._models = {}

    def _add_model(self, name, model):
        self._models[name] = model

    def fit(self):
        for _, model in self._models.items():
            print("\nmodel {} is training".format(_))
            model.fit()

    def predict(self, item_list):
        df_list = []
        for name, model in self._models.items():
            y_rf = model.predict(item_list)
            for item, d in y_rf.items():
                df = pd.DataFrame(columns=['item', 'model', 'current_period', 'forecast_period', 'yhat', 'true'])
                df['forecast_period'] = d['period']
                df['yhat'] = d['yhat']
                df['true'] = d['true']
                df['item'] = item
                df['model'] = name
                df['current_period'] = d['current_period']
                df_list.append(df)
                
            model.close_session()
        
        return pd.concat(df_list)


def build_NN_model(conf, NN_Models, data):
    nn_model = NN_Models()
    for nn in conf:
        name = nn['name']
        params = nn['param']
        nn_model._add_model(name=name, model=reg_models.get(name, None)(data=data, **params))
    return nn_model
