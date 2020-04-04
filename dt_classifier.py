#!usr/bin/env
# -*- coding: utf-8 -*-
'''
Открытый курс машинного обучения. Тема 10. Градиентный бустинг
https://habr.com/ru/company/ods/blog/327250/
'''

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import target
from collections import defaultdict

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

def load_extra(filename, series_name):
    g = pd.read_csv(filename, names=['DATE', 'VALUE'], skiprows=1, index_col=['DATE'], infer_datetime_format=True, date_parser=lambda dt: pd.datetime.strptime(dt, '%Y-%m-%d')).VALUE
    g = (g + (g/np.inf)).fillna(0)
    return g.rename(series_name)
    
def trade(predict, award):
    x = 1.0
    for p, a in zip(predict, award):
        # print('p:{} a:{}'.format(p, a))
        x *= a[p]
    return x

        
def confusion_report(y_true, y_predict):
    m = confusion_matrix(y_true, y_predict)
    print('row=expected  column=predicted')
    print(m)
    
if __name__ == '__main__':
    
    y = target.load_target('1_GAZP.csv', lambda d: target.growth(d, q=3))
    g = load_extra('gazp_best.csv', 'VALUE')
    g0 = load_extra('gazp_best_new_target.csv', 'VALUE0')
    g1 = load_extra('gazp_other_best.csv','OTHER_VALUE')
    g2 = load_extra('gazp_similar_best.csv','SIMILAR_VALUE')
    g3 = load_extra('gazp_similar_best_new_target.csv','SIMILAR_VALUE0')
    g4 = load_extra('gazp_top5_close_high_q4.csv', 'TOP5')
    g5 = load_extra('gazp_imoex_close_high_q4.csv', 'IMOEX')
    
    s = load_series(['1_GAZP.csv'], True)
    x = pd.concat( s , axis=1)
    x = pd.concat( s + [g, g0, g1, g2, g3, g4, g5], axis=1)
    x = x + x/np.inf

    _, x = y.TARGET.align(x, join='left', fill_value=0)
    _, y = y.TARGET.align(y, join='left', fill_value=0)
    x = x.fillna(0)
    
    y_train = y['2009-01-01':'2018-12-31']
    x_train = x['2009-01-01':'2018-12-31']

    y_test = y['2019-01-01':]
    x_test = x['2019-01-01':]
    
    '''
    print('\nmutual information:')
    for c in x_train.columns:
        v = x_train[c]
        v = (v + v/np.inf).fillna(0)  # v/np.inf = 0 или inf/inf = NaN        
        info = mutual_info_classif(v.values.reshape(-1,1), y_train.TARGET, n_neighbors=5, random_state=4838474)[0]
        print('{}: {}'.format(c, info))
    '''   
    
    #clf = DecisionTreeClassifier(random_state=0, min_samples_split=30, max_depth=15, min_samples_leaf=5)
    #clf = RandomForestClassifier(random_state=0, min_samples_split=150, min_samples_leaf=50, n_estimators=500)
    # best configuration for 2-class classification
    #clf = GradientBoostingClassifier(random_state=0, n_estimators=500, learning_rate=0.02, max_depth=1, min_samples_leaf=150)
    # best configuration for 4-class classification
    clf = GradientBoostingClassifier(random_state=0, n_estimators=2000, learning_rate=0.02, max_depth=1, min_samples_leaf=25)
    
    clf.fit(x_train, y_train.TARGET, y_train.WEIGHT)
    print('\nfeature importances:')
    for c, v in sorted(zip(x_train.columns, clf.feature_importances_), key = lambda x: x[1]):
        print('{0}: {1:.4f}'.format(c, v))

    print('\ntrain score:')
    print(clf.score(x_train, y_train.TARGET))
    print('\ntest score:')
    print(clf.score(x_test, y_test.TARGET))
    predict = clf.predict(x_test)
    print('\nconfusion matrix:')
    confusion_report(y_test.TARGET, predict)
    print(classification_report(y_test.TARGET, predict))
    
    
    print('\ntrade score:')
    print(trade(predict, y_test.AWARD))

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