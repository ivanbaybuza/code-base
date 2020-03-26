import sys
from contextlib import contextmanager
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

def result_analyze():
    pass

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pkl.load(file)
    return data

# 显示程序运行时间，结合with使用
@contextmanager
def timeit(label='time'):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))


@contextmanager
def redirect_stdout(new_stdout):
    """
    屏蔽打印消息
    使用示例：
    with open(os.devnull, "w") as file, redirect_stdout(file):
        要屏蔽消息的代码
    """
    saved_stdout, sys.stdout = sys.stdout, new_stdout
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def make_nomalize(data):
    """
    归一化 MAX_MIN
    :param data: dataframe or series or ndarray
    :return: 归一化后的dataframe or series
    """
    if isinstance(data, pd.DataFrame):
        for i in range(data.shape[1]):
            data.iloc[:, i] = (data.iloc[:, i] - data.iloc[:, i].min()) / (
                    data.iloc[:, i].max() - data.iloc[:, i].min())
        return data
    elif isinstance(data, pd.Series):
        return (data - data.min()) / (data.max() - data.min())
    elif isinstance(data, np.ndarray):
        scalery = MinMaxScaler(feature_range=(0, 1))
        return scalery.fit_transform(data)
    else:
        raise ValueError("type of data is not dataframe or series or ndarray")


def make_week_to_date(period):
    """
    求一个周维度对应日期的星期一，速度较慢，不适合序列使用，单独数据可以使用
    :param period:int-->201820
    :return: 这一周的周一的日期
    """
    year = period // 100
    date_range = pd.date_range(start="{}-12-24".format(year - 1), end="{}-01-07".format(year + 1), freq="W-MON")
    se = pd.Series(data=date_range)
    se.index = se.map(lambda x: x.isocalendar()[0] * 100 + x.isocalendar()[1])
    return se.loc[period]


def get_n_period(period, n=1, is_monthly=True):
    """
    目前支持周维度和月维度
    :param period: year_mon int ex: 201808
    :param n: int
    :param is_monthly: int，True:月维度格式，False:周维度格式
    :return: new period int
    """
    # 月维度：获取period的后n个月或者前n个月
    if is_monthly:
        year, mon = divmod(period, 100)
        y, m = divmod(n, 12)
        year += y
        if mon + m > 12:
            year += 1
            mon = mon + m - 12
        elif mon + m < 1:
            year -= 1
            mon = mon + m + 12
        else:
            mon += m
        return year * 100 + mon

    # 周维度：获取period的后n周或者前n周
    else:
        new_date = make_week_to_date(period) + datetime.timedelta(7 * n)
        year, week, _ = new_date.isocalendar()
        return year * 100 + week


def get_date_lead(period1, period2, is_monthly=True):
    """
    获取两个时间序列相差周期，目前支持周维度和月维度
    :param period1:  year_mon or year_week  int ex: 201808
    :param period2:
    :param is_monthly:  bool True:月维度  False：周维度
    :return:  int eg: 1 2
    """
    # 月维度
    if is_monthly:
        year = period1 // 100 - period2 // 100
        mon = period1 % 100 - period2 % 100
        date_lead = year * 12 + mon
        return date_lead

    # 周维度
    else:
        return (make_week_to_date(period1) - make_week_to_date(period2)).days // 7


def get_acc(data, col1, col2):
    """计算准确率：min(v1, v2) / max(v1, v2)"""
    v1 = data[col1]
    v2 = data[col2]
    if max(v1, v2) == 0:
        return np.nan
    if np.isnan(v1) or np.isnan(v2):
        return np.nan
    return round(min(v1, v2) / max(v1, v2), 4)


def restrict_value(data, comp_data):
    """
    约束预测结果，data >= comp_data
    :param data: ndarray
    :param comp_data: ndarray 和data形状相同
    :return: ndarray
    """
    if data.ndim == 1:
        return np.where(data >= comp_data, data, comp_data)
    elif data.ndim == 2:
        arr_list = []
        for col in range(data.shape[1]):
            arr_list.append(np.where(data[:, col] >= comp_data, data[:, col], comp_data))
        return np.vstack(arr_list).T
    else:
        print("the dimensions of data must be 1 or 2, but %d" % data.ndim)
        raise ValueError("the dimensions of array must be 1 or 2")
