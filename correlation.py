#!usr/bin/env
# -*- coding: utf-8 -*-

'''
https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
https://habr.com/ru/post/351610/
'''

import finam.loader as fnm
import json
import codecs
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv


def time_series_rolling_corr(s1, s2, period, min_periods=1):
    values=[]
    for end in s1.index.values:
        start = end - pd.Timedelta(period)
        values.append(s1[start:end].corr(s2, min_periods=min_periods))
    s = pd.Series(values, index=s1.index)
    s.name = 'corr{p} {s1} vs {s2}'.format(s1=s1.name, s2=s2.name, p=period)
    return s
        
    
def best_correlation(s1, s2, max_shift, window, min_periods=0):
    max_corr_abs = 0
    best_ss1 = None
    for shift in range(-max_shift, max_shift+1):
        ss1 = s1.shift(periods=shift)
        shift_str = '-{}'.format(abs(shift)) if shift > 0 else '+{}'.format(abs(shift)) if shift < 0 else ''
        ss1.name = '{s1}{shift}'.format(s1=s1.name, shift=shift_str)
        c = abs(ss1.corr(s2))
        if c > max_corr_abs:
            max_corr_abs = c
            best_ss1 = ss1
            
    best = time_series_rolling_corr(best_ss1, s2, window, min_periods=min_periods)
    print('{} {}'.format(best.name, max_corr_abs))
    return best, max_corr_abs

def read_all():
    items = []
    for filename in os.listdir('finam/data'):
        print('load {}'.format(filename))
        try:
            d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
            ch = (d.CLOSE - d.OPEN)/d.OPEN
            ch.name = d.name
            items.append(ch)
        except Exception as e:
            print('error {}'.format(e))
    return items

def process(items, dir, max_shift, window, min_periods):
    if not os.path.exists(dir):
        os.makedirs(dir)
    best_corr = lambda s1, s2: best_correlation(s1, s2, max_shift=max_shift, window=window, min_periods=min_periods)
    names = [i.name for i in items]
    with codecs.open(os.path.join(dir, 'table.csv'), 'w', 'utf-8') as out:
        total_corr = csv.writer(out, delimiter=',')
        total_corr.writerow(['instrument'] + names)
        for item in items:
            corr = [(name, c, value) for (name, (c, value)) in ((i.name, best_corr(item, i)) for i in items)]
            total_corr.writerow([item.name] + [value for (name,c,value) in corr])
            corr = [(c, value) for (name, c, value) in corr if name != item.name and value > 0.25]
            corr = list(sorted(corr, key=lambda x: x[1], reverse=True))
            if len(corr) > 0:
                n = min(len(corr), 6)
                f,ax=plt.subplots(n+1, 1, figsize=(14, 3*(n+1)))
                for i in range(n):
                    corr[i][0].plot(color='black', grid=True, ax=ax[i])
                    ax[i].legend()
                y = [c[1] for c in corr]
                x =[c[0].name.split(' vs ')[1] for c in corr]
                ax[n].barh(x, y, height=0.5)
                plt.savefig(os.path.join(dir, '{}.png'.format(item.name)))
        

if __name__ == '__main__':
    #instruments = json.loads(codecs.open('finam/instruments/instruments.json', 'r', 'utf-8').read())
    #markets = json.loads(codecs.open('finam/markets.json', 'r', 'utf-8').read())
    
    
    items = read_all()
    last_year_items = [i['2019-01-01':] for i in items]
    process(items, 'result/correlation/all_time_D', max_shift=5, window='90D', min_periods=50)
    process(last_year_items, 'result/correlation/2019_D', max_shift=5, window='60D', min_periods=40)
    
