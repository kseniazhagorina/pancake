#!usr/bin/env
# -*- coding: utf-8 -*-

'''
TARGET - целевая фукнция принимающая значения [0..n]
AWARD - матрица, определяющая какое вознаграждение получает алгоритм если угадает/не угадает целевое значение
'''

import finam.loader as fnm
import json
import codecs
import pandas as pd
import numpy as np
import os
import csv
import information_gain as ig

def load_target(filename, target):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    d = d.shift(-1)
    return target(d)

def split_qcut(s, q=3):
    vals, catind = pd.qcut(s, q=q).factorize(sort=True)
    return vals, catind.categories

def split_thr(s, thr):
    vals = (s > thr).astype(int)
    categories = [pd.Interval(left=-np.inf, right=thr, closed='right'), pd.Interval(left=thr, right=np.inf, closed='neither')]
    return vals, categories    
    
def growth(d, split):
    '''максимальный рост следующего дня по сравнению с закрытием предыдущего'''
    yc = d.CLOSE.shift(1)
    growth = ((d.HIGH - yc)/yc).dropna()
    vals, catind = split(growth)
    print('target: {}'.format(catind))
    
    change = ((d.CLOSE - yc)/yc).dropna()
    g, c = growth.align(change, join='inner')
    
    def _award(predict, gi, ci):
        '''Если предсказывают низкий рост - не покупаем,
           Если фактический рост в этот день (gi) больше чем предсказано - получаем прибыль
           если меньше - фиксируем убыток на конец дня (ci)'''
        if predict.left - 0.002 < 0.003:
            return 1.0
        return 1.0 + (predict.left if gi > predict.left else ci) - 0.002              
    
        
    award = [[_award(p, gi, ci) for p in catind] for gi, ci in zip(g, c)]
    got = np.array([a[p] for p,a in zip(vals, award)])
    lost = np.array([sum(ai for i,ai in enumerate(a) if i != p)/(len(a)-1) for p,a in zip(vals,award)])
    w1 = np.array([max(0, g-1.0) for g in got])
    w2 = np.array([max(0, min(1.0-l, 0.05)) for l in lost])
    w3 = np.array([1.0]*len(got))
    
    w1 = w1/w1.sum()*len(got)
    w2 = w2/w2.sum()*len(got)
    
    
    weight = w1 + w2 + 2.0*w3
    weight = weight/weight.sum()*len(weight)
    print('w1: {}..{}'.format(w1.min(), w1.max()))
    print('w2: {}..{}'.format(w2.min(), w2.max()))
    for i in range(len(catind)):
        print('{}: {}'.format(i, sum(wi for vi,wi in zip(vals, weight) if vi == i)))
    
    
    
    return pd.concat([pd.Series(vals, index=growth.index).rename('TARGET'),
                      pd.Series(award, index=g.index).rename('AWARD'),
                      pd.Series(weight, index=g.index).rename('WEIGHT')], axis=1)
                      

def change(d, split):
    yc = d.CLOSE.shift(1)
    c = ((d.CLOSE - yc)/yc).dropna()
    vals, catind = split(c)
    print('target: {}'.format(catind))
    
    award = [[1.0]*len(catind)]*len(vals)
    weight = [1.0]*len(vals)
    for i in range(len(catind)):
        print('{}: {}'.format(i, sum(wi for vi,wi in zip(vals, weight) if vi == i)))
    
    return pd.concat([pd.Series(vals, index=c.index).rename('TARGET'),
                      pd.Series(award, index=c.index).rename('AWARD'),
                      pd.Series(weight, index=c.index).rename('WEIGHT')], axis=1)
    
    
                      
    
def load_series(filenames, addf=False, adddt=False):
    series = []
    for filename in filenames:
        print('load {}'.format(filename))
        try:
            d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
            if addf:
                a = ig.add_factors(d)
                dt = ig.add_datetime_factors(d) if adddt else None
                name = d.name
                d = pd.concat([d, a, dt], axis=1)
                d.name = name
            series += [d[c].rename(d.name+'.'+d[c].name) for c in d.columns]
        except Exception as e:
            print('error {}'.format(e))            
    return series
    
    