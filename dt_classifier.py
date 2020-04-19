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


def load_extra(filename, series_name):
    g = pd.read_csv(filename, names=['DATE', 'VALUE'], skiprows=1, index_col=['DATE'], infer_datetime_format=True, date_parser=lambda dt: pd.datetime.strptime(dt, '%Y-%m-%d')).VALUE
    g = (g + (g/np.inf)).fillna(0)
    return g.rename(series_name)
    
def trade(predict, award):
    x = 1.0
    for p, a in zip(predict, award):
        x *= a[p]
    return x

        
def confusion_report(y_true, y_predict):
    m = confusion_matrix(y_true, y_predict)
    print('row=expected  column=predicted')
    print(m)
    
if __name__ == '__main__':
    
    y = target.load_target('1_GAZP.csv', lambda d: target.growth(d, split=lambda s: target.split_qcut(s, q=4)))
    g = load_extra('gazp_best.csv', 'VALUE')
    g015 = load_extra('gazp_close_high_015.csv', 'VALUE015')
    #g006 = load_extra('gazp_close_high_006.csv', 'VALUE006')
    #g002 = load_extra('gazp_close_close_002.csv', 'VALUE002')
    g1 = load_extra('gazp_other_best.csv','OTHER_VALUE')
    g2 = load_extra('gazp_similar_best.csv','SIMILAR_VALUE')
    g21 = load_extra('gazp_similar_best_new_target.csv','SIMILAR_VALUE0')
    g201 = load_extra('gazp_similar_close_high_01.csv','SIMILAR_VALUE01')
    g4 = load_extra('gazp_top5_close_high_q4.csv', 'TOP5')
    g5 = load_extra('gazp_top6_close_high_015.csv', 'TOP6')
    #g6 = load_extra('gazp_imoex_close_high_02.csv', 'IMOEX')
    
    s = target.load_series(['1_GAZP.csv'], True, True)
    x = pd.concat( s , axis=1)
    x = pd.concat( s + [g, g015, g1, g2, g21, g4, g5], axis=1)
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
    # clf = GradientBoostingClassifier(random_state=0, n_estimators=500, learning_rate=0.02, max_depth=1, min_samples_leaf=50)
    # best configurations for 3-class classification
    # clf = GradientBoostingClassifier(random_state=0, n_estimators=1000, learning_rate=0.005, max_depth=2, min_samples_leaf=25)
    #clf = GradientBoostingClassifier(random_state=0, n_estimators=2500, learning_rate=0.015, max_depth=1, min_samples_leaf=25)
    # best configurations for 4-class classification
    clf = GradientBoostingClassifier(random_state=0, n_estimators=3000, learning_rate=0.005, max_depth=1, min_samples_leaf=25)
    #clf = GradientBoostingClassifier(random_state=0, n_estimators=2000, learning_rate=0.002, max_depth=2, min_samples_leaf=25)
    
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

