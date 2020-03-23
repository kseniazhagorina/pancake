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
    vals, catind = pd.qcut(growth, q=q).factorize()
    
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
        
        
    award = [[_award(p, gi, ci) for p in catind] for gi, ci in zip(g, c)]
    
    return pd.concat([pd.Series(vals, index=growth.index).rename('TARGET'),
                      pd.Series(award, index=g.index).rename('AWARD')], axis=1)
                      
                      
