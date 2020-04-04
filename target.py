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

def load_target(filename, target):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    d = d.shift(-1)
    return target(d)    
    
def growth(d, q=3):
    '''максимальный рост следующего дня по сравнению с закрытием предыдущего'''
    yc = d.CLOSE.shift(1)
    growth = ((d.HIGH - yc)/yc).dropna()
    vals, catind = pd.qcut(growth, q=q).factorize(sort=True)
    print('target: {}'.format(catind))
    
    change = ((d.CLOSE - yc)/yc).dropna()
    g, c = growth.align(change, join='inner')
    
    def _award(predict, gi, ci):
        '''Если предсказывают низкий рост - не покупаем,
           Если фактический рост в этот день (gi) больше чем предсказано - получаем прибыль
           если меньше - фиксируем убыток на конец дня (ci)'''
        a = predict.left - 0.002
        if a < 0.005:
            return 1.0
        return 1.0 + (a if gi > predict.left else ci) - 0.002              
        
        
    award = [[_award(p, gi, ci) for p in catind.categories] for gi, ci in zip(g, c)]
    got = np.array([a[p] for p,a in zip(vals, award)])
    lost = np.array([sum(ai for i,ai in enumerate(a) if i != p)/(len(a)-1) for p,a in zip(vals,award)])
    w1 = np.array([max(0, g-1.0) for g in got])
    w2 = np.array([max(0, 1.0-l) for l in lost])
    w3 = np.array([1.0]*len(got))
    
    w1 = w1/w1.sum()*len(got)
    w2 = w2/w2.sum()*len(got)
    
    weight = 0.4*w1 + w2 + w3
    weight = weight/weight.sum()*len(weight)
    print('w1: {}..{}'.format(w1.min(), w1.max()))
    print('w2: {}..{}'.format(w2.min(), w2.max()))
    for i in range(len(catind)):
        print('{}: {}'.format(i, sum(wi for vi,wi in zip(vals, weight) if vi == i)))
    
    
    
    return pd.concat([pd.Series(vals, index=growth.index).rename('TARGET'),
                      pd.Series(award, index=g.index).rename('AWARD'),
                      pd.Series(weight, index=g.index).rename('WEIGHT')], axis=1)
                      
                      
