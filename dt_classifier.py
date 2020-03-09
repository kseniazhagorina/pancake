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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
        
def load_target(filename):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    d = d.shift(-1)
    growth = (d.HIGH - d.OPEN)/d.OPEN
    change = (d.CLOSE - d.OPEN)/d.OPEN

    target = growth.apply(lambda x: 1 if x > 0.008 else 0) # цель - рост на 0.8% в день
    # weight = change.apply(lambda x: 3.0 if x > 0.008 else 1.0 if x > 0.002 else 2.0 if x > -0.008 else 3.0)
    weight = change.apply(lambda x: 1.0)
    
    return pd.concat([target.rename('TARGET'), weight.rename('WEIGHT'), growth.rename('GROWTH'), change.rename('CHANGE')], axis=1).dropna()
   
    
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

def trade(predict, growth, change):
    x = 1.0
    for p, g, c in zip(predict, growth, change):
        if p:
            res = 0.008 if g > 0.008 else c
            x *= (1.0 + res - 0.002)
    return x
    
if __name__ == '__main__':
    
    y = load_target('1_GAZP.csv')
    g = pd.read_csv('gazp_best.csv', names=['DATE', 'VALUE'], skiprows=1, index_col=['DATE'], infer_datetime_format=True, date_parser=lambda dt: pd.datetime.strptime(dt, '%Y-%m-%d')).VALUE
    g1 = pd.read_csv('gazp_other_best.csv', names=['DATE', 'OTHER_VALUE'], skiprows=1, index_col=['DATE'], infer_datetime_format=True, date_parser=lambda dt: pd.datetime.strptime(dt, '%Y-%m-%d')).OTHER_VALUE
       
    s = load_series(['1_GAZP.csv'], False)
    x = pd.concat( [item for item in s if item.name in ['GAZP.VLT', 'GAZP.VOLR']] + [g] + [g1], axis=1)
    _, x = y.TARGET.align(x, join='left', fill_value=0)
    
    y_train = y['2009-01-01':'2018-12-31']
    x_train = x['2009-01-01':'2018-12-31']
  
    y_test = y['2019-01-01':]
    x_test = x['2019-01-01':]
    
    print('\nmutual information:')
    for c in x_train.columns:
        v = x_train[c]
        v = v + v/np.inf  # v/np.inf = 0 или inf/inf = NaN        
        info = mutual_info_classif(v.values.reshape(-1,1), y_train.TARGET, n_neighbors=5, random_state=4838474)[0]
        print('{}: {}'.format(c, info))
       
    
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=30, max_depth=15, min_samples_leaf=5)
    clf = RandomForestClassifier(random_state=0, min_samples_split=30, max_depth=15, min_samples_leaf=5, n_estimators=150)
    clf.fit(x_train, y_train.TARGET, y_train.WEIGHT)
    print('\nfeature importances:')
    for c, v in zip(x_train.columns, clf.feature_importances_):
        print('{0}: {1:.4f}'.format(c, v))

    print('\ntrain score:')
    print(clf.score(x_train, y_train.TARGET, y_train.WEIGHT))
    print('\ntest score:')
    print(clf.score(x_test, y_test.TARGET, y_test.WEIGHT))
    
    predict = clf.predict(x_test)
    print('\ntrade score:')
    print(trade(predict, y_test.GROWTH, y_test.CHANGE))
    print(trade([1.0]*len(predict), y_test.GROWTH, y_test.CHANGE))
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
    
    
    
'''
load 1_GAZP.csv

mutual information:
GAZP.VOLR: 0.0034650049982394293
GAZP.VLT: 0.007512356126658126
VALUE: 0.090676580979145
OTHER_VALUE: 0.08194905177353817

feature importances:
GAZP.VOLR: 0.2278
GAZP.VLT: 0.2834
VALUE: 0.3082
OTHER_VALUE: 0.1806

train score:
0.8042264752791068

test score:
0.6135458167330677

trade score:
0.8422167338722348 # торговать пока рановато =)
0.6579993608744921
'''    