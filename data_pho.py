# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : xzy
# @FILE     : data_pho.py
# @Time     : 2021/5/14 10:43
# @Software : PyCharm
import kwargs as kwargs
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# 读取csv保存的数据
fig_title=0
for daily in pd.read_csv('D:\pythonProject\StockDL\indo10csv\ASII.JK_testing.csv',
                    index_col=0,
                    parse_dates=True,
                    usecols=[0, 1, 2, 3, 4, 6],
                    chunksize=100
                    ):
    fig_title+=1
    mpf.plot(daily,
             type='candle',
             volume=True,
             style='yahoo',
             figratio=(20, 20),
             figscale=5,
             savefig="A_stock-%d_candle_line.png" % fig_title
             )




# from skimage import io,data
# img=data.chelsea()
# io.imshow(img)
# io.imsave(‘D:\pythonProject\StockDL\dataset\‘,png)

