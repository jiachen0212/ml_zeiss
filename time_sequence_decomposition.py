# coding=utf-8

# STL 预测季节数据
# https://zhuanlan.zhihu.com/p/267541046

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.datasets import co2

register_matplotlib_converters()
data = co2.load(True).data
data = data.resample('M').mean().ffill()

from statsmodels.tsa.seasonal import STL

res = STL(data).fit()
res.plot()
plt.show()
