# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : xzy
# @FILE     : data_pho.py
# @Time     : 2021/5/13 8:43
# @Software : PyCharm

import mplfinance as mpf
import pandas as pd
import glob

path=r'indo10csv'
filenames = glob.glob(path + "/*.csv")
fig_title=0
for filename in filenames:
    for daily in pd.read_csv(filename,
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
             savefig="dataset/A_stock-%d_candle_line.png" % fig_title
             )
