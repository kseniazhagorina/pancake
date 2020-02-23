#!usr/bin/env
# -*- coding: utf-8 -*-

'''
https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
https://habr.com/ru/post/351610/
'''

import finam.loader as fnm
import json
import codecs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
from sklearn.feature_selection import mutual_info_classif
     

def add_factors(d):
    change = (d.CLOSE - d.OPEN)/d.OPEN
    growth = (d.HIGH - d.OPEN)/d.OPEN
    first = lambda x: ([i for i in x if i != np.nan] or [np.nan])[0]
    open_1w_ago = d.OPEN.rolling('7D', min_periods=5).apply(first)
    open_2w_ago = d.OPEN.rolling('14D', min_periods=10).apply(first)
    open_1m_ago = d.OPEN.rolling('30D', min_periods=20).apply(first)
    open_3m_ago = d.OPEN.rolling('90D', min_periods=60).apply(first)
    change_1w = (d.CLOSE - open_1w_ago)/open_1w_ago
    change_2w = (d.CLOSE - open_2w_ago)/open_2w_ago
    change_1m = (d.CLOSE - open_1m_ago)/open_1m_ago
    change_3m = (d.CLOSE - open_3m_ago)/ open_3m_ago
    
    avg5 = d.AVG.rolling(window=5).mean()
    avg10 = d.AVG.rolling(window=10).mean()
    avg20 = d.AVG.rolling(window=20).mean()
    # экспоненциальное скользящее среднее y^[i] = (1-alpha)*y^[i-1] + alpha*y[i]
    ewm04 = d.AVG.ewm(alpha=0.4).mean()
    ewm02 = d.AVG.ewm(alpha=0.2).mean()
    vlt5 = (d.AVG - ewm02).rolling(window=5).std()
    vlt10 = (d.AVG - ewm02).rolling(window=10).std()
    
    return pd.concat([
        change.rename('CHANGE'),
        change.shift(1).rename('CHANGE-1'),
        change.shift(2).rename('CHANGE-2'),
        growth.rename('GROWTH'),
        growth.shift(1).rename('GROWTH-1'),
        growth.shift(2).rename('GROWTH-2'),
        change_1w.rename('CHANGE_1W'),
        change_2w.rename('CHANGE_2W'),
        change_1m.rename('CHANGE_1M'),
        change_3m.rename('CHANGE_3M'),
        
        #avg5.rename('AVG5'),
        #avg10.rename('AVG10'),
        #avg20.rename('AVG20'),
        #ewm04.rename('EWM04'), 
        #ewm02.rename('EWM02'),
        
        vlt5.rename('VLT5'),
        vlt10.rename('VLT10'),
        (d.VLT/d.OPEN).rename('VLTP'),
        (vlt5/d.OPEN).rename('VLT5P'),
        (vlt10/d.OPEN).rename('VLT10P'),
        
        # пересечение границы ценового коридора вверх и вниз
        (((ewm02 + 2*vlt10) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_2VLT10'),
        ((d.AVG - (ewm02 - 2*vlt10))/d.AVG).rename('AVG_CROSS_DOWN_2VLT10'),
        (((ewm02 + 3*vlt10) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_3VLT10'),
        ((d.AVG - (ewm02 - 3*vlt10))/d.AVG).rename('AVG_CROSS_DOWN_3VLT10'),
        (((ewm02 + 2*vlt5) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_2VLT5'),
        ((d.AVG - (ewm02 - 2*vlt5))/d.AVG).rename('AVG_CROSS_DOWN_2VLT5'),
        (((ewm02 + 3*vlt5) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_3VLT5'),
        ((d.AVG - (ewm02 - 3*vlt5))/d.AVG).rename('AVG_CROSS_DOWN_3VLT5'),
        
        (((ewm02 + 2*vlt10) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_2VLT10'),
        ((d.LOW - (ewm02 - 2*vlt10))/d.LOW).rename('LOW_CROSS_DOWN_2VLT10'),
        (((ewm02 + 3*vlt10) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_3VLT10'),
        ((d.LOW- (ewm02 - 3*vlt10))/d.LOW).rename('LOW_CROSS_DOWN_3VLT10'),
        (((ewm02 + 2*vlt5) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_2VLT5'),
        ((d.LOW - (ewm02 - 2*vlt5))/d.LOW).rename('LOW_CROSS_DOWN_2VLT5'),
        (((ewm02 + 3*vlt5) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_3VLT5'),
        ((d.LOW - (ewm02 - 3*vlt5))/d.LOW).rename('LOW_CROSS_DOWN_3VLT5')
        
        
        ],
        axis=1
    )


if __name__ == '__main__':
    #instruments = json.loads(codecs.open('finam/instruments/instruments.json', 'r', 'utf-8').read())
    #markets = json.loads(codecs.open('finam/markets.json', 'r', 'utf-8').read())
    
    filename = '1_GAZP.csv'
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    a = add_factors(d)
    target = a.GROWTH.apply(lambda x: int(x > 0.008)).shift(-1) # цель - рост на 0.8% в день    
    a1 = pd.concat([pd.concat([d, a], axis=1).fillna(100500), target.rename('TARGET')], axis=1).dropna()
    info = mutual_info_classif(a1, a1.TARGET.values)
    for col, val in zip(a1.columns, info):
        print(col, val)

'''        
OPEN 0.004359224837064524
HIGH 0.008169738701871543
LOW 0.0
CLOSE 0.0001832821046225952
VOL 0.019028172723636105
AVG 0.002226366638127697
VLT 0.025402566866148613
CHANGE 0.0315656718156625
CHANGE-1 0.03466452730194347
CHANGE-2 0.004619166195846791
GROWTH 0.02388404070372907
GROWTH-1 0.01650179388913875
GROWTH-2 0.009602215441460782
CHANGE_1W 0.020300979235526206
CHANGE_2W 0.007128797754013538
CHANGE_1M 0.023095417982881727
CHANGE_3M 0.014086878261797331
VLT5 0.024129733405265696
VLT10 0.020954120503821727
VLTP 0.03620861217403637
VLT5P 0.018856128228981195
VLT10P 0.0029923080686760084
AVG_CROSS_UP_2VLT10 0.013612526351780785
AVG_CROSS_DOWN_2VLT10 0.016757426482836735
AVG_CROSS_UP_3VLT10 0.015572206424085966
AVG_CROSS_DOWN_3VLT10 0.017576537391149838
AVG_CROSS_UP_2VLT5 0.016357391587655146
AVG_CROSS_DOWN_2VLT5 0.014177588942060337
AVG_CROSS_UP_3VLT5 0.002926770120636135
AVG_CROSS_DOWN_3VLT5 0.01850323869840942
HIGH_CROSS_UP_2VLT10 0.023636200178887234
LOW_CROSS_DOWN_2VLT10 0.015431279997867975
HIGH_CROSS_UP_3VLT10 0.019235195950511708
LOW_CROSS_DOWN_3VLT10 0.014728620802308745
HIGH_CROSS_UP_2VLT5 0.008663719811536241
LOW_CROSS_DOWN_2VLT5 0.02084590830997257
HIGH_CROSS_UP_3VLT5 0.023828374644657924
LOW_CROSS_DOWN_3VLT5 0.009710234182474187
TARGET 0.6916426399331562
'''