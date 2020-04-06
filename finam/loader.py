#!usr/bin/env
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import read_csv
import numpy as np
import os.path

def read(filename):
    d = pd.read_csv(filename, 
             names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'],
             parse_dates={'DATETIME': ['DATE', 'TIME']},
             index_col=['DATETIME'],
             infer_datetime_format=True)
    d.name = os.path.splitext(os.path.basename(filename))[0].split('_', maxsplit=1)[1]  
    return d
    
def avgcross(x):
    x = ((x.ffill().bfill() - x.mean()) > 0)
    return sum(x.ffill().bfill() ^ x.shift(1).ffill().bfill())

def resample(data, period, dropna=True):
    params = {'rule': period, 'closed':'left', 'label':'left'}
    # data.OPEN.filter(data.VOL > 0).resample(**params).mean() работает очень медленно
    vol_filter = lambda s: s + ((data.VOL-data.VOL)/data.VOL) # 0 или NaN
    
    close = vol_filter(data.CLOSE).resample(**params).last().rename('CLOSE')
    open = vol_filter(data.OPEN).resample(**params).first().rename('OPEN')
    high = vol_filter(data.HIGH).resample(**params).max().rename('HIGH')
    low = vol_filter(data.LOW).resample(**params).min().rename('LOW')
    #low1 = vol_filter(data.LOW).resample(**params).apply(
    #        lambda x: np.nan if len(x) == 0 else np.min(x[0:np.argmax(x)+1])).rename('LOW1')
    vol = data.VOL.resample(**params).sum()
    
    
    volr = (data.VOL * data.OPEN).resample(**params).sum().rename('VOLR')  # объем в рублях
    avg = (volr/vol).rename('AVG')
    vlt = vol_filter(data.OPEN).resample(**params).std().rename('VLT') # волатильность - стандатное отклонение цены от средней
    avgc = vol_filter(data.OPEN).resample(**params).apply(avgcross).rename('AVGC')
    
    d = pd.concat([open, high, low, close, vol, volr, avg, avgc, vlt], axis=1)
    d.name = data.name
    if dropna:
        d.dropna(inplace=True)
        return d
    else:
        # замещаем дни когда биржа не работала ценами на закрытие последнего дня
        d.CLOSE.fillna(method='pad', inplace=True)
        d.OPEN.fillna(d.CLOSE, inplace=True)
        d.HIGH.fillna(d.CLOSE, inplace=True)
        d.LOW.fillna(d.CLOSE, inplace=True)
        d.AVG.fillna(d.CLOSE, inplace=True)
        d.VLT.fillna(0, inplace=True)
        
        return d