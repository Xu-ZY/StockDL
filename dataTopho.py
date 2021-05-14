# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : xzy
# @FILE     : data_pho.py
# @Time     : 2021/5/13 8:43
# @Software : PyCharm

import mplfinance as mpf
import pandas as pd

daily=pd.read_csv('D:\pythonProject\StockDL\indo10csv\ASII.JK_testing.csv',
                    index_col=0,
                    parse_dates=True,
                    usecols=[0, 1, 2, 3, 4, 6],
                    chunksize=5
                    )
for piece in daily:
    for i in range(1000):
        i += 1
    mpf.plot(daily,
             type='candle',
             volume=True,
             style='yahoo',
             figratio=(20, 20),
             figscale=5,
             savefig="A_stock-%d_candle_line.png" %i
             )