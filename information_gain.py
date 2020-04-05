#!usr/bin/env
# -*- coding: utf-8 -*-

'''
https://habr.com/ru/post/351610/
https://habr.com/ru/company/ods/blog/327242/
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
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from scipy.optimize import minimize
import statsmodels.api as sm

        

# https://habr.com/ru/company/ods/blog/327242/
# level: l[x] = a*s[x] + (1-a)(y[x-1])   
# trend: t[x] = b*(l[x] - l[x-1]) + (1-b)*t[x-1]
# y[x] = l[x] + t[x])
def ewm2(series, alpha, beta):
    level = [0]*len(series)
    trend = [0]*len(series)
    y = [0]*len(series)
    y[0] = level[0] = series[0]
    for x in range(1, len(series)):
        s = y[x] if series.values[x] == np.nan else series.values[x]
        level[x] = alpha*s + (1 - alpha)*y[x-1]
        trend[x] = beta*(level[x] - level[x-1]) + (1-beta)*trend[x-1]
        y[x] = level[x] + trend[x]
    return pd.Series(y, index=series.index)



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
    
    low1w = d.LOW.rolling('6D').min().shift(1)
    high1w = d.HIGH.rolling('6D').max().shift(1)
    
    # экспоненциальное скользящее среднее y^[i] = (1-alpha)*y^[i-1] + alpha*y[i]
    ewm04 = d.AVG.ewm(alpha=0.4).mean()
    
    

    avgewm02 = d.AVG.ewm(alpha=0.2).mean()
    openewm02 = d.OPEN.ewm(alpha=0.2).mean()
    highewm02 = d.HIGH.ewm(alpha=0.2).mean()
    closeewm02 = d.CLOSE.ewm(alpha=0.2).mean()
    
    vlt5 = (d.AVG - avgewm02.shift(1)).rolling(window=5).std()
    vlt10 = (d.AVG - avgewm02.shift(1)).rolling(window=10).std()
    vlt5p = ((d.AVG - avgewm02.shift(1))/d.AVG).rolling(window=5).std()
    vlt10p = ((d.AVG - avgewm02.shift(1))/d.AVG).rolling(window=10).std()
    
    openewm2 = ewm2(d.OPEN, 0.2, 0.2)
    avgewm2 = ewm2(d.AVG, 0.2, 0.2)
    highewm2 = ewm2(d.HIGH, 0.2, 0.2)
    lowewm2 = ewm2(d.LOW, 0.2, 0.2)
    

    return pd.concat([
        change.rename('CHANGE'),
        change.shift(1).rename('CHANGE-1'),
        change.shift(2).rename('CHANGE-2'),
        change.rolling(window=5).mean().rename('CHANGE5'),
        growth.rename('GROWTH'),
        growth.shift(1).rename('GROWTH-1'),
        growth.shift(2).rename('GROWTH-2'),
        growth.rolling(window=5).mean().rename('GROWTH5'),
        growth.rolling(window=5).max().rename('MAX_GROWTH5'),
        growth.rolling(window=3).max().rename('MAX_GROWTH3'),
        growth.rolling(window=5).min().rename('MIN_GROWTH5'),
        growth.rolling(window=3).min().rename('MIN_GROWTH3'),
        change_1w.rename('CHANGE_1W'),
        change_2w.rename('CHANGE_2W'),
        change_1m.rename('CHANGE_1M'),
        change_3m.rename('CHANGE_3M'),
        
        ((d.HIGH - d.LOW)/d.LOW).rename('HIGH_LOW_P'),
        ((d.HIGH - d.LOW)/d.LOW).rolling(window=5).mean().rename('HIGH_LOW_P5'),
        
        ((avgewm02 - d.CLOSE)/d.CLOSE).rename('AVG_EWM02_CLOSE'),
        ((highewm02 - d.CLOSE)/d.CLOSE).rename('HIGH_EWM02_CLOSE'),
        ((highewm02 - openewm02)/openewm02).rename('HIGH_EWM02_OPEN_EWM02'),
        
        vlt5.rename('VLT5'),
        vlt10.rename('VLT10'),
        
        (d.VLT/d.OPEN).rename('VLTP'),
        vlt5p.rename('VLT5P'),
        vlt10p.rename('VLT10P'),
        
        # пересечение границы ценового коридора вверх и вниз
        (((avgewm02 + 2*vlt10) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_2VLT10'),
        ((d.AVG - (avgewm02 - 2*vlt10))/d.AVG).rename('AVG_CROSS_DOWN_2VLT10'),
        (((avgewm02 + 3*vlt10) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_3VLT10'),
        ((d.AVG - (avgewm02 - 3*vlt10))/d.AVG).rename('AVG_CROSS_DOWN_3VLT10'),
        (((avgewm02 + 2*vlt5) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_2VLT5'),
        ((d.AVG - (avgewm02 - 2*vlt5))/d.AVG).rename('AVG_CROSS_DOWN_2VLT5'),
        (((avgewm02 + 3*vlt5) - d.AVG)/d.AVG).rename('AVG_CROSS_UP_3VLT5'),
        ((d.AVG - (avgewm02 - 3*vlt5))/d.AVG).rename('AVG_CROSS_DOWN_3VLT5'),
        
        (((avgewm02 + 2*vlt10) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_2VLT10'),
        ((d.LOW - (avgewm02 - 2*vlt10))/d.LOW).rename('LOW_CROSS_DOWN_2VLT10'),
        (((avgewm02 + 3*vlt10) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_3VLT10'),
        ((d.LOW- (avgewm02 - 3*vlt10))/d.LOW).rename('LOW_CROSS_DOWN_3VLT10'),
        (((avgewm02 + 2*vlt5) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_2VLT5'),
        ((d.LOW - (avgewm02 - 2*vlt5))/d.LOW).rename('LOW_CROSS_DOWN_2VLT5'),
        (((avgewm02 + 3*vlt5) - d.HIGH)/d.HIGH).rename('HIGH_CROSS_UP_3VLT5'),
        ((d.LOW - (avgewm02 - 3*vlt5))/d.LOW).rename('LOW_CROSS_DOWN_3VLT5'),
        
        openewm02.rename('OPEN_EWM02'),
        openewm2.rename('OPEN_EWM2'),
        d.OPEN.rolling(window=5).mean().rename('OPEN5'),
        
        ((d.AVG - low1w)/(high1w - low1w)).rename('AVG_OVER_1W'),
        ((d.LOW - low1w)/(high1w - low1w)).rename('LOW_OVER_1W'),
        ((d.HIGH - low1w)/(high1w - low1w)).rename('HIGH_OVER_1W'),
        
        ((openewm2 - openewm02)/openewm02).rename('OPEN_EWM2_EWM02'),
        ((avgewm2 - avgewm02)/avgewm02).rename('AVG_EWM2_EWM02'),
        ((highewm2 - highewm02)/highewm02).rename('HIGH_EWM2_EWM02'),

        ],
        axis=1
    )
    
def add_datetime_factors(d):
    dayofweek = pd.Series([date.dayofweek for date in d.index], index=d.index)
    day = pd.Series([date.day for date in d.index], index=d.index)
    month = pd.Series([date.month for date in d.index], index=d.index)
    
    return pd.concat([
        dayofweek.rename('DAYOFWEEK'),
        day.rename('DAY'),
        month.rename('MONTH')
        ],
        axis=1
    )
    
    
def mean_squared_error(y, y_pred):
    df = pd.concat([y.rename('Y'), y_pred.rename('Y_PRED')], axis=1).dropna()
    return sk_mean_squared_error(df.Y, df.Y_PRED)


if __name__ == '__main__':
    #instruments = json.loads(codecs.open('finam/instruments/instruments.json', 'r', 'utf-8').read())
    #markets = json.loads(codecs.open('finam/markets.json', 'r', 'utf-8').read())
    
    filename = '1_GAZP.csv'
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    a = add_factors(d)
    dt = add_datetime_factors(d)
    target = a.GROWTH.apply(lambda x: int(x > 0.008)).shift(-1) # цель - рост на 0.8% в день    
    a1 = pd.concat([pd.concat([d, a, dt], axis=1).fillna(100500), target.rename('TARGET')], axis=1).dropna()
    info = mutual_info_classif(a1, a1.TARGET.values)
    for col, val in sorted(zip(a1.columns, info), key=lambda x: x[1], reverse=True):
        print('{0}\t{1:.4f}'.format(col, val))
        
    '''
    OPEN    0.0072
    HIGH    0.0095
    LOW     0.0029
    CLOSE   0.0008
    VOL     0.0190
    AVG     0.0022
    VLT     0.0254
    CHANGE  0.0315
    CHANGE-1        0.0352
    CHANGE-2        0.0046
    CHANGE5 0.0037
    GROWTH  0.0232
    GROWTH-1        0.0138
    GROWTH-2        0.0118
    GROWTH5 0.0272
    CHANGE_1W       0.0219
    CHANGE_2W       0.0000
    CHANGE_1M       0.0218
    CHANGE_3M       0.0144
    AVG_EWM02_CLOSE 0.0138
    HIGH_EWM02_CLOSE        0.0112
    HIGH_EWM02_OPEN_EWM02   0.0199
    VLT5    0.0229
    VLT10   0.0236
    VLTP    0.0363
    VLT5P   0.0167
    VLT10P  0.0163
    AVG_CROSS_UP_2VLT10     0.0102
    AVG_CROSS_DOWN_2VLT10   0.0060
    AVG_CROSS_UP_3VLT10     0.0313
    AVG_CROSS_DOWN_3VLT10   0.0232
    AVG_CROSS_UP_2VLT5      0.0000
    AVG_CROSS_DOWN_2VLT5    0.0241
    AVG_CROSS_UP_3VLT5      0.0205
    AVG_CROSS_DOWN_3VLT5    0.0202
    HIGH_CROSS_UP_2VLT10    0.0219
    LOW_CROSS_DOWN_2VLT10   0.0275
    HIGH_CROSS_UP_3VLT10    0.0291
    LOW_CROSS_DOWN_3VLT10   0.0130
    HIGH_CROSS_UP_2VLT5     0.0166
    LOW_CROSS_DOWN_2VLT5    0.0255
    HIGH_CROSS_UP_3VLT5     0.0000
    LOW_CROSS_DOWN_3VLT5    0.0231
    OPEN_EWM02      0.0201
    OPEN_EWM2       0.0129
    OPEN_EWM2_EWM02 0.0000
    AVG_EWM2_EWM02  0.0229
    HIGH_EWM2_EWM02 0.0167
    TARGET  0.6912
    '''
        
    
    v = d.AVG[0:-500]
    error = lambda x: mean_squared_error(v, ewm2(v, *x).shift(1)) # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    opt = minimize(error, x0=[0.2, 0.2], method="TNC", bounds = ((0, 1), (0, 1)))
    v = d.AVG[-500:]
    print('ewm2: a=0.2 b=0.2: {}'.format(error([0.2, 0.2])))
    print('ewm2: a={0:.3f} b={1:.3f}: {2}'.format(opt.x[0], opt.x[1], error(opt.x)))
    print('ewm: a=0.2: {}'.format(mean_squared_error(v, v.ewm(alpha=0.2).mean().shift(1))))
    # a=0.2 b=0.2: 5.625125988932227
    # a=1.000 b=0.000: 2.829266273661487
    # значит просто модель плохо предсказыает, так что оптимизация свелась просто к предсказанию "завтра будет как вчера"
    
    '''
    https://ru.wikipedia.org/wiki/Стационарность
    Стационарность или постоянство — свойство процесса не менять свои характеристики со временем.
    Стационарный процесс — это стохастический процесс, у которого не изменяется распределение вероятности при смещении во времени.
    Следовательно, такие параметры, как среднее значение и дисперсия.
    '''
    print('Dicket-Fuller test on stationarity: p-value < 0.01 mean that series is stationary') 
    print('OPEN: p-value: {0:.3f}'.format(sm.tsa.stattools.adfuller(d.OPEN)[1]))
    print('CHANGE: p-value: {0:.3f}'.format(sm.tsa.stattools.adfuller(a.CHANGE)[1]))
    print('GROWTH: p-value: {0:.3f}'.format(sm.tsa.stattools.adfuller(a.GROWTH)[1]))
    print('OPEN[x] - OPEN[x-1]: p-value: {0:.3f}'.format(sm.tsa.stattools.adfuller((d.OPEN - d.OPEN.shift(1)).dropna())[1]))
    

    d.OPEN.plot(color='black')
    d.OPEN.shift(1).plot(color='blue') # предсказание завтра будет как вчера
    a.OPEN5.shift(1).plot(color='red') # среднее по последним 5 дням
    a.OPEN_EWM2.shift(1).plot(color='yellow') # экспоненциальное срденее с a=0.2 b=0.2
    plt.show()

