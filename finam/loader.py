#!usr/bin/env
# -*- coding: utf-8 -*-

import pandas as pd

from pandas import read_csv

def read(filename):
    d = pd.read_csv(filename, 
             names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'],
             parse_dates={'DATETIME': ['DATE', 'TIME']},
             index_col=['DATETIME'],
             infer_datetime_format=True)


def resample(data, period, dropna=False):
    params = {'rule': period, 'closed':'left', 'label':'left'}
    close = data.CLOSE.resample(**params).last()
    open = data.OPEN.resample(**params).first()
    high = data.HIGH.resample(**params).max()
    low = data.LOW.resample(**params).min()
    vol = data.VOL.resample(**params).sum()
    
    d = pd.concat([open, high, low, close, vol], axis=1)
    if dropna:
        d.dropna(inplace=True)
        return d
    else:
        # замещаем дни когда биржа не работала ценами на закрытие последнего дня
        d.CLOSE.fillna(method='pad', inplace=True)
        d.OPEN.fillna(d.CLOSE, inplace=True)
        d.HIGH.fillna(d.CLOSE, inplace=True)
        d.LOW.fillna(d.CLOSE, inplace=True)
        return d