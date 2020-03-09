#!usr/bin/env
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import read_csv
import os.path

def read(filename):
    d = pd.read_csv(filename, 
             names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'],
             parse_dates={'DATETIME': ['DATE', 'TIME']},
             index_col=['DATETIME'],
             infer_datetime_format=True)
    d.name = os.path.splitext(os.path.basename(filename))[0].split('_', maxsplit=1)[1]  
    return d

def resample(data, period, dropna=True):
    params = {'rule': period, 'closed':'left', 'label':'left'}
    close = data.CLOSE.resample(**params).last()
    open = data.OPEN.resample(**params).first()
    high = data.HIGH.resample(**params).max()
    low = data.LOW.resample(**params).min()
    vol = data.VOL.resample(**params).sum()
    # data.OPEN.filter(data.VOL > 0).resample(**params).mean() работает очень медленно
    vol_filter = lambda s: s + ((data.VOL-data.VOL)/data.VOL) # 0 или NaN
    avg = vol_filter(data.OPEN).resample(**params).mean()
    avg.name = 'AVG'
    volr = (data.VOL * data.OPEN).resample(**params).sum()  # объем в рублях
    volr.name = 'VOLR'
    vlt = vol_filter(data.OPEN).resample(**params).std() # волатильность - стандатное отклонение цены от срдней за сутки (тогда когда были продажи)
    vlt.name = 'VLT'

    
    d = pd.concat([open, high, low, close, vol, volr, avg, vlt], axis=1)
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