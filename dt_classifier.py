#!usr/bin/env
# -*- coding: utf-8 -*-

import sys
import finam.loader as fnm
import json
import codecs
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import information_gain as ig
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
        
def load_target(filename):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    growth = (d.HIGH - d.OPEN)/d.OPEN
    target = growth.apply(lambda x: int(x > 0.008)).shift(-1).dropna() # цель - рост на 0.8% в день
    return target.rename('TARGET')
    
def load_series(filenames, addf=False):
    series = []
    for filename in filenames:
        print('load {}'.format(filename))
        d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
        if addf:
            a = ig.add_factors(d)
            name = d.name
            d = pd.concat([d, a], axis=1)
            d.name = name
        series += [d[c].rename(d.name+'.'+d[c].name) for c in d.columns]    
    return series

if __name__ == '__main__':
    
    y = load_target('1_GAZP.csv')
    g = pd.read_csv('gazp_best.csv', names=['DATE', 'VALUE'], index_col=['DATE'], infer_datetime_format=True)
    s = load_series(['1_GAZP.csv'], False)
    x = pd.concat( s + [g.VALUE], axis=1).fillna(0)
    
    
    y_train = y['2009-01-01':'2018-12-31']
    x_train = x['2009-01-01':'2018-12-31']
    y_train, x_train = y_train.align(x_train, join='left')
    print('\nmutual_info_classif:')
    mi = mutual_info_classif(x_train, y_train, n_neighbors=5, random_state=4838474)
    for c, v in zip(x_train.columns, mi):
        print('{0}: {1:.4f}'.format(c, v))
    
    
    y_test = y['2019-01-01':]
    x_test = x['2019-01-01':]
    y_test, x_test = y_test.align(x_test, join='left')
    
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=50, max_depth=15)
    clf.fit(x_train, y_train)
    print('\nfeature importances:')
    for c, v in zip(x_train.columns, clf.feature_importances_):
        print('{0}: {1:.4f}'.format(c, v))

    print('\ntrain score:')
    print(clf.score(x_train, y_train))
    print('\ntest score:')
    print(clf.score(x_test, y_test))
    
'''
D:\Development\Python\pancake>python dt_classifier.py
load 1_GAZP.csv

mutual_info_classif:
GAZP.VOLR: 0.0035
GAZP.VLT: 0.0105
GAZP.HIGH_LOW_P5: 0.0317
VALUE: 0.0901

feature importances:
GAZP.VOLR: 0.1656
GAZP.VLT: 0.2735
GAZP.HIGH_LOW_P5: 0.3699
VALUE: 0.1910

train score:
0.722488038277512

test score:
0.5258964143426295

D:\Development\Python\pancake>python dt_classifier.py
load 1_GAZP.csv

mutual_info_classif:
GAZP.VOLR: 0.0035
GAZP.VLT: 0.0105
VALUE: 0.0904

feature importances:
GAZP.VOLR: 0.2457
GAZP.VLT: 0.3663
VALUE: 0.3880

train score:
0.6925837320574163

test score:
0.601593625498008

VALUE - фактор полученный генетическим программированием из базовых значений 1_GAZP.csv
GAZP.HIGH_LOW_P5 выглядит как полезная фича, но приводит к переобучению
Обязательно нужна кросс-валидация! и подбор оптимальных факторов
Кроме того умение угадывать в 60% случаев вообще по факту ничего не значит, т.к. важна цена ошибки.
'''
    