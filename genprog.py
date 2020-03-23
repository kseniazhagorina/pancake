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
from datetime import datetime, timedelta
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import minimize
import statsmodels.api as sm

import information_gain as ig
from correlation import time_series_rolling_corr

import gc

np.random.seed(838373)

def random_choice(x, p=None):
    return x[np.random.choice(len(x), p=p)]

class ValueType:
    SERIES = 'SERIES'
    INT = 'INT'
    FLOAT = 'FLOAT'
    NONE = 'NONE'
    
class Argument:
    def __init__(self, name, value_type, interval=None):
        self.name = name
        self.value_type = value_type
        self.interval = interval # for INT, FLOAT
        
    def __repr__(self):
        return self.value_type + ('({})'.format(self.interval) if self.interval is not None else '')
        
class Func:
    def __init__(self, name, arguments, result_type, calc):
        self.name = name
        self.arguments = arguments
        self.result_type = result_type
        self.calc = calc
        self.signature = '(' + ','.join([str(a) for a in self.arguments]) + ')->' + self.result_type
   
def rolling(name, calc):
    return  Func('rolling.{}'.format(name),
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('window', ValueType.INT, interval=pd.Interval(left=1, right=31, closed='both'))],
            result_type=ValueType.SERIES,
            calc = calc)

ROLLING_MEAN = rolling('mean', lambda x, window: x.rolling(window=window).mean())
ROLLING_MIN = rolling('min', lambda x, window: x.rolling(window=window).min())
ROLLING_MAX = rolling('max', lambda x, window: x.rolling(window=window).max())
ROLLING_STD = rolling('std', lambda x, window: x.rolling(window=window).std())
ROLLING_VAR = rolling('var', lambda x, window: x.rolling(window=window).var())
ROLLING_SUM = rolling('sum', lambda x, window: x.rolling(window=window).sum())
def minmax(x, window): # изменение в сравнении с min/max предыдущего периода
    minx = x.rolling(window=window).min().shift(1).ffill()
    maxx = x.rolling(window=window).max().shift(1).ffill()
    diff = maxx - minx
    return (x - minx)/diff * (diff/diff) 
MINMAX = rolling('minmax', lambda x, window: minmax(x, window))

ROLLING_CORR = Func('rolling.corr',
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('y', ValueType.SERIES),
                       Argument('window', ValueType.INT, interval=pd.Interval(left=5, right=365, closed='both'))],
            result_type=ValueType.SERIES,
            calc = lambda x, y, window: time_series_rolling_corr(x, y, period='{}D'.format(window)))
            
def shift(name, calc):
    return  Func('sp.{}'.format(name),
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('periods', ValueType.INT, interval=pd.Interval(left=1, right=31, closed='both'))],
            result_type=ValueType.SERIES,
            calc = calc)
SHIFT = shift('shift', lambda x, periods: x.shift(periods))
CHANGE = shift('change', lambda x, periods: x.pct_change(periods))


def series_op(name, calc):
    return  Func('ss.{}'.format(name),
            arguments=[Argument('x1', ValueType.SERIES),
                       Argument('x2', ValueType.SERIES)],
            result_type=ValueType.SERIES,
            calc = calc)

SSUM = series_op('sum', lambda x1, x2: x1 + x2)
SSUB = series_op('sub', lambda x1, x2: x1 - x2)
SMUL = series_op('mul', lambda x1, x2: x1 * x2)
SDIV = series_op('div', lambda x1, x2: x1 / x2 * (x2/x2)) # x2/x2 = 1 или NaN чтобы не возникало деления на 0
SPCNT = series_op('pct', lambda x1, x2: (x1 - x2) / x1 * (x1/x1))
SMAX = series_op('max', lambda x1, x2: (x1.gt(x2)).astype(float)*x1 + (x1.lt(x2)).astype(float)*x2)
SMIN = series_op('min', lambda x1, x2: (x1.gt(x2)).astype(float)*x2 + (x1.lt(x2)).astype(float)*x1)
SLT = series_op('lt', lambda x1, x2: (x1.lt(x2)).astype(float))
SAND = series_op('and', lambda x1, x2: ((x1 > 0) & (x2 > 0)).astype(float))
SOR = series_op('or', lambda x1, x2: ((x1 > 0) | (x2 > 0)).astype(float))
SXOR = series_op('xor', lambda x1, x2: ((x1 > 0) ^ (x2 > 0)).astype(float))

def series_unary_op(name, calc):
    return  Func('s.{}'.format(name),
            arguments=[Argument('x', ValueType.SERIES)],
            result_type=ValueType.SERIES,
            calc = calc)

SNOT = series_unary_op('not', lambda x: (x <= 0).astype(float))
SABS = series_unary_op('abs', lambda x: x.abs())
SQRT = series_unary_op('sqrt', lambda x: np.sqrt(x))
SQR = series_unary_op('square', lambda x: x * x)
SLOG = series_unary_op('log', lambda x: x.apply(lambda v: math.log(v) if v > 0 else np.nan))
CUMSUM = series_unary_op('cumsum', lambda x: x.cumsum())
CUMMIN = series_unary_op('cummin', lambda x: x.cummax())
CUMMAX = series_unary_op('cummax', lambda x: x.cummin())
FFILLNA = series_unary_op('ffill', lambda x: x.fillna(method='ffill'))
FILLNA0 = series_unary_op('fill0', lambda x: x.fillna(value=0))


def series_num_op(name, calc):
    return  Func('sc.{}'.format(name),
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('c', ValueType.FLOAT)],
            result_type=ValueType.SERIES,
            calc = calc)

NSUM = series_num_op('sum', lambda x, c: x + c)
NMUL = series_num_op('mul', lambda x, c: x * c)
NLT = series_num_op('lt', lambda x, c: (x < c).astype(float))
NGT = series_num_op('gt', lambda x, c: (x > c).astype(float))

EWM = Func('sf.ewm',
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('a', ValueType.FLOAT, interval=pd.Interval(left=0, right=1, closed='both'))],
            result_type=ValueType.SERIES,
            calc = lambda x, a: x.ewm(a).mean())
EWM2 = Func('sf.ewm2',
            arguments=[Argument('x', ValueType.SERIES),
                       Argument('a', ValueType.FLOAT, interval=pd.Interval(left=0, right=1, closed='both')),
                       Argument('b', ValueType.FLOAT, interval=pd.Interval(left=0, right=1, closed='both'))],
            result_type=ValueType.SERIES,
            calc = lambda x, a, b: ig.ewm2(x, a, b))


ALL_OPERATIONS = [ROLLING_MEAN, ROLLING_MIN, ROLLING_MAX, ROLLING_STD, ROLLING_VAR, ROLLING_SUM, #ROLLING_CORR,
                  SSUM, SSUB, SMUL, SDIV, SPCNT, SMAX, SMIN, SLT, SAND, SOR, SXOR,
                  SNOT, SABS, SQRT, SQR, SLOG, CUMMIN, CUMMAX, NSUM, NMUL, NLT, NGT, # CUMSUM
                  EWM, EWM2, SHIFT, CHANGE, MINMAX, FFILLNA, FILLNA0]


                           
class Node:
    def __init__(self):
        self.size = 0
        self.height = 0
        self.value_type = ValueType.NONE
        
    @property
    def children(self):
        return []
    
    @property
    def value(self):
        return None
        
    def get_child(self, path):
        if len(path) == 0:
            return self
        return self.children[path[0]].get_child(path[1:])
    
    def get_argument(self, path):
        if len(path) == 0:
            # текущий узел - корень
            return Argument('x', value_type=self.value_type) 
        node = self.get_child(path[:-1])
        return node.func.arguments[path[-1]]
        
class FuncNode(Node):
    def __init__(self, func, values):     
        self.func = func
        self.__children = values
        self.size = sum(c.size for c in self.__children) + 1
        self.height = max(c.height for c in self.__children) + 1
        self.value_type = self.func.result_type
        self.__value = None
        
    @property
    def value(self):
        if self.__value is not None:
            return self.__value

        all_valid = True
        for i, c in enumerate(self.__children):
            argument = self.func.arguments[i]
            if c.value_type != argument.value_type:
                raise Exception("inconsistant value type: need {} but was {}".format(self.func.arguments[i].value_type, c.value_type))
            if c.value is None:
                all_valid = False
            if argument.interval and c.value is not None and c.value not in argument.interval:
                all_valid = False
                
        if all_valid:
            try:
                values = (c.value for c in self.__children)
                self.__value = self.func.calc(*values)
            except Exception as e:
                sys.stderr.write('{}\n{}\n'.format(e, print_node(self)))
        return self.__value
        
        
    @property
    def children(self):
        return self.__children

class LeafNode(Node):
    def __init__(self, value_type, value):
        self.value_type = value_type
        self.__value = value
        self.size = 1
        self.height = 1

    @property
    def value(self):
        return self.__value

def replace(node, path, replacement):
    if len(path) == 0:
        return replacement
    children = list(node.children)
    child = children[path[0]]
    children[path[0]] = replace(child, path[1:], replacement)
    return FuncNode(node.func, children)
    
def print_node(node, prefix='', intent=''):
    s = ''
    if isinstance(node, FuncNode):
        s += intent + prefix + node.func.name + '\n'
        for i, arg in enumerate(node.func.arguments):
            s += print_node(node.children[i], prefix='{}='.format(arg.name), intent=intent+'  ')
    elif isinstance(node, LeafNode):
        if node.value_type == ValueType.SERIES:
            s += intent + prefix + node.value.name + '\n'
        else:
            s += intent + prefix + str(node.value) + '\n'
    return s            
            
            

    
class GenProg:
    def __init__(self, operations, series, target, n, p_series=None, max_node_size=None, max_n=None, best_n=None):    
        self.__operations = operations
        self.__series = [LeafNode(ValueType.SERIES, s) for s in series]
        self.__target = target
        self.__count = n
        self.__max_count = max_n or n*3
        self.__best_count = best_n or n//3
        self.__items = []
        self.epoch = 0
        self.__last_epoch_inc_score = 0
        self.__last_epoch_elapsed_time = timedelta(0)
        p = np.array(p_series or [1.0]*len(series))
        self.__p_series = p/p.sum()
        self.__max_node_size = max_node_size
    
    @property
    def best(self):
        return self.__items[0][0]
    
    def start(self):
        hash = set()
        self.__items = []
        for i in range(self.__count*3):
            item = self.generate()
            s = print_node(item)
            if s not in hash:
               value = self.eval(item)
               if value != -np.inf:
                   self.__items.append((item, value))
                   if len(self.__items) == self.__count:
                       break
        
    def print_state(self):
        
        best, score = self.__items[0]
        if self.__last_epoch_inc_score == self.epoch:
            print(print_node(best, intent='    '))

        print('\n')
        print('epoch: {}'.format(self.epoch))
        print('    finished at: {}   elapsed: {}'.format(datetime.now(), self.__last_epoch_elapsed_time))
        print('    avg: {}'.format(sum(x[1] for x in self.__items)/len(self.__items)))
        print('    avg size: {}'.format(sum(x[0].size for x in self.__items)/len(self.__items)))
        print('    last inc score at: {}'.format(self.__last_epoch_inc_score))
        print('    best: {} (height:{} size:{})'.format(score, best.height, best.size))
        print('\n')

           
    def get_random_subtree_path(self, node, max_len=None):
        '''return: path to subtree, [] mean self'''
        rnd = np.random.random()
        p_i = 0
        if max_len is None or max_len > 0:
            for i in range(len(node.children)):
                p_i += float(node.children[i].size)/node.size
                if rnd < p_i:
                    return [i] + self.get_random_subtree_path(node.children[i], max_len-1 if max_len is not None else None)
        return []
        
    def cross(self, a, b):
        '''случайное поддерево в a заменяется на случайное поддерево в b'''
        for _ in range(100):
            patha = self.get_random_subtree_path(a)
            node_a = a.get_child(patha)
            pathb = self.get_random_subtree_path(b)
            node_b = b.get_child(pathb)
            if node_a.value_type == node_b.value_type:
                return replace(a, patha, node_b)
        return None

    def mutate_node_on_similar_func(self, node):
        similar = [f for f in self.__operations if f.signature == node.func.signature and f.name != node.func.name]
        if similar:
            new_node = FuncNode(random_choice(similar), node.children)
            return new_node
        return None

    def mutate_node_on_child(self, node):
        path = self.get_random_subtree_path(node)
        if path:
            child = node.get_child(path, max_len=2)
            if child.value_type == node.value_type:
                return child
        return None
        
    def mutate_node_on_extra_node(self, node):
        if node.value_type == ValueType.SERIES:
            new = self.generate(h=2)
            children = [i for i, a in enumerate(new.func.arguments) if a.value_type == ValueType.SERIES]
            if children:
                return replace(new, [i], node)
        return None
        
    def mutate_node_on_leaf_node(self, node):
        if node.value_type == ValueType.SERIES:
            return random_choice(self.__series, p=self.__p_series)
        return None
        
    
    def mutate_leaf(self, node, argument):
        if node.value_type == ValueType.SERIES:
            return random_choice(self.__series, p=self.__p_series)
    
        if node.value_type == ValueType.INT or node.value_type == ValueType.FLOAT:
            for _ in range(100):
                value = node.value + node.value * np.random.normal() # случайно увеличиваем или уменьшаем имеющееся значение
                value = int(round(value, 0)) if node.value_type == ValueType.INT else value
                if value == node.value:
                    continue
                if argument and argument.interval and value not in argument.interval:
                    continue
                return LeafNode(node.value_type, value)
                
        return None
    
    def random_leaf(self, argument):
        if argument.value_type == ValueType.SERIES:
            return random_choice(self.__series, p=self.__p_series)
        if argument.value_type == ValueType.INT or argument.value_type == ValueType.FLOAT:
            if argument.interval:
                left = argument.interval.left if argument.interval.left != -math.inf else 100000*(np.random.power(1) - 1.0)
                right = argument.interval.right if argument.interval.right != math.inf else -100000*(np.random.power(1) - 1.0)
                if argument.value_type == ValueType.INT:
                    left = int(left)
                    if argument.interval.open_left:
                        left += 1
                    if argument.interval.closed_right:
                        right += 1
                    return LeafNode(argument.value_type, np.random.randint(low=left, high=right))
                if argument.value_type == ValueType.FLOAT:
                    if argument.interval.open_left:
                        left += 0.00001
                    if argument.interval.closed_right:
                        right += 0.00001
                    
                    value = np.random.random()*(right - left) + left
                    return LeafNode(argument.value_type, value)
            value = np.random.normal() * 100000
            if argument.value_type == ValueType.INT:
                value = round(value, 0)
            return LeafNode(argument.value_type, value)
                                
             
    def mutate(self, a):
        patha = self.get_random_subtree_path(a)
        node = a.get_child(patha)
        for _ in range(100):
            new_node = None
            rnd = np.random.random()
            if isinstance(node, FuncNode):
                new_node = (self.mutate_node_on_similar_func(node) if rnd < 0.33 else
                            self.mutate_node_on_child(node) if rnd < 0.33 else
                            self.mutate_node_on_extra_node(node) if rnd < 0.33 else
                            self.mutate_node_on_leaf_node(node))
            elif isinstance(node, LeafNode):
                argument = a.get_argument(patha)
                new_node = (self.mutate_leaf(node, argument) if rnd < 0.66 else
                            self.mutate_node_on_extra_node(node) if rnd < 0.33 else
                            self.random_leaf(argument))
                    
            if new_node and node.value_type == new_node.value_type:
                return replace(a, patha, new_node) 
        return None

        
    def generate(self, h=3):
        func = random_choice(self.__operations)
        values = [self.generate(h-1) if a.value_type == ValueType.SERIES and h > 2 else self.random_leaf(a) for a in func.arguments]
        return FuncNode(func, values)
        
    def select_best_child(self, node):
        if node.value is None or node.value_type != ValueType.SERIES:
            return None, -np.inf
        bestv = self.eval(node)
        best = node
        for c in node.children:
            bc, bv = self.select_best_child(c)
            if bv > bestv:
                bestv = bv
                best = bc
        return best, bestv
        
    def eval(self, node):
        
        # бесполезное прибавление числа и умножение на число в корне дерева ничего не меняет
        if isinstance(node, FuncNode) and node.func.name in [NSUM.name, NMUL.name, SABS.name]:
            return -np.inf
            
        if self.__max_node_size is not None and node.size > self.__max_node_size:
            return -np.inf

        if node.value is None:
            return -np.inf    
       
        v = node.value
        v = (v + v/np.inf).fillna(0)  # v/np.inf = 0 или inf/inf = NaN        
        v, t = v.align(self.__target, join='right', fill_value = 0)
               
        try:
            v = v.values.reshape(-1, 1)
            info = mutual_info_classif(v, self.__target, n_neighbors=5, random_state=4838474)[0]
            # return info - 0.0002 * node.height # - 0.000002 * node.size
            return info - 0.00001 * node.size
        except Exception as e:
            sys.stderr.write('{}'.format(e))
            return -np.inf      


    
    
    def next_epoch(self):

        epoch_start = datetime.now()
        prev_best_score = self.__items[0][1]
        next = self.__items[0:3]   # 3 лучших переходят в следующую эпоху как есть
        nexthash = set(print_node(x[0]) for x in next)
            
        p = np.fromiter((v for x, v in self.__items), float, len(self.__items))
        p = (p - p.min())**2
        p = p/p.sum()
        
        while len(next) < self.__max_count:
            rnd = np.random.random()
            x, _ = random_choice(self.__items, p=p)
            y, _ = random_choice(self.__items)
            new = self.cross(x, y) if rnd < 0.5 else self.cross(y, x)
            for i in range(np.random.choice(4, p=[0.5, 0.25, 0.125, 0.125])):
                if new:
                    new = self.mutate(new)
            if new is not None and new.value is not None:
                hash = print_node(new)
                if hash not in nexthash:
                    value = self.eval(new)
                    if value != -np.inf:
                        next.append((new, value))
                        nexthash.add(hash)
        
        self.epoch += 1
        # 25% результата займут лучшие 75% - просто случайные
        inds = np.array(sorted(range(len(next)), key=lambda ind: next[ind][1], reverse=True))
        n_best = self.__best_count
        rest = np.random.choice(len(next) - n_best, self.__count - n_best, replace=False) + n_best # индексы случайных элементов next[n_best:] 
        self.__items = [next[inds[i]] for i in range(n_best)] + [next[inds[i]] for i in rest]
 
        if self.__items[0][1] != prev_best_score:
            self.__last_epoch_inc_score = self.epoch
        
        self.__last_epoch_elapsed_time = datetime.now() - epoch_start
        
def load_target(filename):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    target = ((d.VOLR/d.VOL).pct_change(1) > 0).astype(float).shift(-1).dropna()
    
    #growth = (d.HIGH - d.OPEN)/d.OPEN
    #change = (d.CLOSE - d.OPEN)/d.OPEN
    #target = ((growth > 0.008) & (change > 0.002)).astype(float).shift(-1).dropna()
    #target = (growth > 0.008).astype(float).shift(-1).dropna() # цель - рост на 0.8% в день
    #target = growth.apply(lambda x: np.round(x)).shift(-1).dropna() # с шагом 1%
    return target
    
def load_series(filenames):
    series = []
    for filename in filenames:
        print('load {}'.format(filename))
        d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
        series += [d[c].rename(d.name+'.'+d[c].name) for c in d.columns]    
    return series
        
        
def run(start, end, target_filename, series_filenames, n, max_epoch, save_as, p_series=None): 
 
    target = load_target(target_filename)[start:end]
    series = load_series(series_filenames)
    ps = [p_series[s.name] if s.name in p_series else min(p_series) for s in series] if p_series else None
    
    # при N = max_node_size * max_n затраты памяти будут примерно N*100 Кб
    g = GenProg(ALL_OPERATIONS, series, target, n, p_series=ps, max_n=n*2, max_node_size=120)
    g.start()
    g.print_state()
    while max_epoch is None or g.epoch < max_epoch:
        g.next_epoch()
        g.print_state()
        g.best.value.to_csv(save_as)
        if g.epoch % 20 == 0:
            gc.collect()
        
   
    
def calcstat(start, end, target_filename, series_filenames):
    target = load_target(target_filename)[start:end]
    series = load_series(series_filenames)
    g = GenProg(ALL_OPERATIONS, series, target, 100)
    
    from collections import defaultdict
    individual = defaultdict(list)
    mutual = defaultdict(list)
    
    def get_child_series(node):
        if isinstance(node, LeafNode) and node.value_type == ValueType.SERIES:
            return [node.value.name]
        result = []
        for c in node.children:
            result += get_child_series(c)
        return result
    
    uniq = set()    
        
    for i in range(10000):
        if i%100 == 0:
            print(i)
        node = g.generate(h=3)
        s = print_node(node)
        if s in uniq:
            continue
        uniq.add(s)
        children = get_child_series(node)
        uniq_children = set(children)
        v = g.eval(node)
        if v == -np.inf:
            print(s)
            continue
        for c in uniq_children:
            individual[c].append(v)
            for c1 in uniq_children:
                if c != c1 or children.count(c) > 1:
                    mutual[(c,c1)].append(v)
                    
    individual_mean = list(sorted([(s, np.mean(v)) for s, v in individual.items()], key=lambda x: x[1], reverse=True))
    individual_top10 = list(
        sorted([
           (s, np.mean(list(sorted(v, reverse=True))[0:10]))
           for s, v in individual.items()
        ], key=lambda x: x[1], reverse=True))
   
    print('individual mean:')
    for s, v in individual_mean:
        print('{0:.4f} {1}'.format(v, s))
    print('individual top10 mean:')
    for s, v in individual_top10:
        print('{0:.4f} {1}'.format(v, s))

    mutual_mean = list(sorted([(s, np.mean(v)) for s, v in mutual.items()], key=lambda x: x[1], reverse=True))
    mutual_top10 = list(
        sorted([
           (s, np.mean(list(sorted(v, reverse=True))[0:10]))
           for s, v in mutual.items()
        ], key=lambda x: x[1], reverse=True))

    print('mutual mean:')
    for s, v in mutual_mean[0:20]:
        print('{0:.4f} {1}'.format(v, s))
    print('mutual top10 mean:')
    for s, v in mutual_top10[0:20]:
        print('{0:.4f} {1}'.format(v, s))        
    

p = [
(0.0389,'SIBN.CLOSE'),
(0.0352,'NVTK.CLOSE'),
(0.0341,'LKOH.CLOSE'),
(0.0339,'GAZP.VOL'),
(0.0339,'GAZP.VLT'),
(0.0338,'LKOH.VOL'),
(0.0335,'NVTK.AVG'),
(0.0331,'LKOH.VLT'),
(0.0330,'LKOH.HIGH'),
(0.0323,'BZ.VOLR'),
(0.0323,'ROSN.OPEN'),
(0.0322,'VTBR.CLOSE'),
(0.0321,'SIBN.LOW'),
(0.0321,'VTBR.VOL'),
(0.0320,'LKOH.VOLR'),
(0.0319,'VTBR.OPEN'),
(0.0318,'GMKN.LOW'),
(0.0318,'GMKN.HIGH'),
(0.0311,'SIBN.VLT'),
(0.0310,'GAZP.HIGH'),
(0.0308,'SIBN.OPEN'),
(0.0306,'SIBN.VOL'),
(0.0306,'ROSN.VOLR'),
(0.0304,'NVTK.VOLR'),
(0.0304,'GMKN.AVG'),
(0.0303,'NVTK.LOW'),
(0.0301,'NVTK.VLT'),
(0.0300,'BZ.AVG'),
(0.0298,'ROSN.AVG'),
(0.0298,'ROSN.HIGH'),
(0.0298,'ROSN.CLOSE'),
(0.0297,'VTBR.VLT'),
(0.0297,'LKOH.LOW'),
(0.0295,'GMKN.CLOSE'),
(0.0292,'GMKN.VOL'),
(0.0290,'NVTK.HIGH'),
(0.0290,'SIBN.HIGH'),
(0.0290,'GAZP.CLOSE'),
(0.0289,'BZ.HIGH'),
(0.0289,'VTBR.LOW'),
(0.0288,'GAZP.VOLR'),
(0.0286,'VTBR.VOLR'),
(0.0286,'LKOH.OPEN'),
(0.0286,'GAZP.OPEN'),
(0.0284,'GMKN.VLT'),
(0.0284,'LKOH.AVG'),
(0.0281,'NG.CLOSE'),
(0.0281,'SIBN.AVG'),
(0.0281,'SIBN.VOLR'),
(0.0280,'VTBR.AVG'),
(0.0280,'GAZP.LOW'),
(0.0279,'NVTK.VOL'),
(0.0278,'VTBR.HIGH'),
(0.0276,'NG.VLT'),
(0.0271,'NVTK.OPEN'),
(0.0271,'ROSN.VOL'),
(0.0270,'ROSN.LOW'),
(0.0269,'GMKN.OPEN'),
(0.0268,'GMKN.VOLR'),
(0.0267,'BZ.CLOSE'),
(0.0264,'NG.VOLR'),
(0.0262,'BZ.VLT'),
(0.0261,'ROSN.VLT'),
(0.0261,'NG.LOW'),
(0.0257,'BZ.OPEN'),
(0.0255,'BZ.VOL'),
(0.0246,'NG.OPEN'),
(0.0244,'NG.VOL'),
(0.0241,'BZ.LOW'),
(0.0235,'NG.AVG'),
(0.0234,'GAZP.AVG'),
(0.0226,'NG.HIGH'),
(0.0451,'NG.LOW'),
(0.0451,'VTBR.VOL'),
(0.0390,'SIBN.VOL'),
(0.0390,'SIBN.CLOSE'),
(0.0331,'VTBR.CLOSE'),
(0.0331,'NVTK.HIGH'),
(0.0301,'SIBN.VLT'),
(0.0301,'BZ.VOLR'),
(0.0288,'GMKN.AVG'),
(0.0288,'LKOH.HIGH'),
(0.0262,'SIBN.LOW'),
(0.0262,'VTBR.AVG'),
(0.0256,'LKOH.OPEN'),
(0.0256,'NVTK.VOLR'),
(0.0255,'NG.OPEN'),
(0.0255,'GMKN.LOW'),
(0.0247,'GAZP.OPEN'),
(0.0247,'GAZP.VOL'),
(0.0246,'GMKN.CLOSE'),
(0.0246,'LKOH.VLT')
]

P_SERIES = {}
for v, name in p:
    if name not in P_SERIES:
        P_SERIES[name] = 0
    P_SERIES[name] += v
    
def main1():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_GAZP.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_best_new_target.csv')
        
def main2():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_GAZP.csv', '1_LKOH.csv', '1_ROSN.csv',
        '1_VTBR.csv', '1_GMKN.csv', '1_NVTK.csv',
        '1_SIBN.csv', '24_NG.csv', '24_BZ.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_other_best_new_target.csv',
        p_series=P_SERIES)
        
def main3():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_LKOH.csv', '1_ROSN.csv', '1_NVTK.csv', '1_SIBN.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_similar_best_new_target.csv')


        
def main_calcstat():
    calcstat('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_GAZP.csv', '1_LKOH.csv', '1_ROSN.csv',
        '1_VTBR.csv', '1_GMKN.csv', '1_NVTK.csv',
        '1_SIBN.csv', '24_NG.csv', '24_BZ.csv'])

    
    
if __name__ == '__main__':
    #instruments = json.loads(codecs.open('finam/instruments/instruments.json', 'r', 'utf-8').read())
    #markets = json.loads(codecs.open('finam/markets.json', 'r', 'utf-8').read())

    main3()

'''
individual mean:
0.0090 SIBN.CLOSE
0.0080 NVTK.AVG
0.0078 NVTK.CLOSE
0.0077 GAZP.VOL
0.0075 SIBN.OPEN
0.0071 NVTK.OPEN
0.0070 SIBN.LOW
0.0069 ROSN.CLOSE
0.0068 SIBN.VOLR
0.0066 ROSN.OPEN
0.0065 LKOH.HIGH
0.0064 SIBN.HIGH
0.0064 LKOH.CLOSE
0.0063 GMKN.LOW
0.0062 NVTK.LOW
0.0062 NVTK.HIGH
0.0062 LKOH.VOL
0.0062 ROSN.LOW
0.0059 LKOH.OPEN
0.0059 GMKN.VOL
0.0059 GAZP.VLT
0.0058 GMKN.HIGH
0.0058 LKOH.LOW
0.0057 BZ.AVG
0.0056 GMKN.AVG
0.0056 ROSN.AVG
0.0056 GAZP.HIGH
0.0056 VTBR.OPEN
0.0056 SIBN.AVG
0.0056 VTBR.HIGH
0.0055 GMKN.CLOSE
0.0055 LKOH.AVG
0.0054 VTBR.VOLR
0.0054 VTBR.AVG
0.0054 NVTK.VOLR
0.0054 BZ.HIGH
0.0054 ROSN.HIGH
0.0053 GAZP.OPEN
0.0053 ROSN.VOLR
0.0053 LKOH.VLT
0.0052 SIBN.VLT
0.0052 BZ.OPEN
0.0052 GMKN.OPEN
0.0051 VTBR.LOW
0.0051 VTBR.CLOSE
0.0050 GAZP.CLOSE
0.0049 NG.VOL
0.0048 ROSN.VLT
0.0048 ROSN.VOL
0.0047 NG.CLOSE
0.0047 SIBN.VOL
0.0047 NG.OPEN
0.0047 BZ.CLOSE
0.0046 GMKN.VOLR
0.0046 GAZP.AVG
0.0046 GAZP.LOW
0.0046 VTBR.VOL
0.0045 GAZP.VOLR
0.0044 NVTK.VLT
0.0044 NVTK.VOL
0.0044 VTBR.VLT
0.0042 LKOH.VOLR
0.0042 GMKN.VLT
0.0041 BZ.LOW
0.0040 BZ.VLT
0.0040 NG.LOW
0.0040 BZ.VOL
0.0039 BZ.VOLR
0.0038 NG.VLT
0.0037 NG.VOLR
0.0033 NG.AVG
0.0032 NG.HIGH

individual top10 mean:
0.0389 SIBN.CLOSE
0.0352 NVTK.CLOSE
0.0341 LKOH.CLOSE
0.0339 GAZP.VOL
0.0339 GAZP.VLT
0.0338 LKOH.VOL
0.0335 NVTK.AVG
0.0331 LKOH.VLT
0.0330 LKOH.HIGH
0.0323 BZ.VOLR
0.0323 ROSN.OPEN
0.0322 VTBR.CLOSE
0.0321 SIBN.LOW
0.0321 VTBR.VOL
0.0320 LKOH.VOLR
0.0319 VTBR.OPEN
0.0318 GMKN.LOW
0.0318 GMKN.HIGH
0.0311 SIBN.VLT
0.0310 GAZP.HIGH
0.0308 SIBN.OPEN
0.0306 SIBN.VOL
0.0306 ROSN.VOLR
0.0304 NVTK.VOLR
0.0304 GMKN.AVG
0.0303 NVTK.LOW
0.0301 NVTK.VLT
0.0300 BZ.AVG
0.0298 ROSN.AVG
0.0298 ROSN.HIGH
0.0298 ROSN.CLOSE
0.0297 VTBR.VLT
0.0297 LKOH.LOW
0.0295 GMKN.CLOSE
0.0292 GMKN.VOL
0.0290 NVTK.HIGH
0.0290 SIBN.HIGH
0.0290 GAZP.CLOSE
0.0289 BZ.HIGH
0.0289 VTBR.LOW
0.0288 GAZP.VOLR
0.0286 VTBR.VOLR
0.0286 LKOH.OPEN
0.0286 GAZP.OPEN
0.0284 GMKN.VLT
0.0284 LKOH.AVG
0.0281 NG.CLOSE
0.0281 SIBN.AVG
0.0281 SIBN.VOLR
0.0280 VTBR.AVG
0.0280 GAZP.LOW
0.0279 NVTK.VOL
0.0278 VTBR.HIGH
0.0276 NG.VLT
0.0271 NVTK.OPEN
0.0271 ROSN.VOL
0.0270 ROSN.LOW
0.0269 GMKN.OPEN
0.0268 GMKN.VOLR
0.0267 BZ.CLOSE
0.0264 NG.VOLR
0.0262 BZ.VLT
0.0261 ROSN.VLT
0.0261 NG.LOW
0.0257 BZ.OPEN
0.0255 BZ.VOL
0.0246 NG.OPEN
0.0244 NG.VOL
0.0241 BZ.LOW
0.0235 NG.AVG
0.0234 GAZP.AVG
0.0226 NG.HIGH
mutual mean:
0.0451 ('NG.LOW', 'VTBR.VOL')
0.0451 ('VTBR.VOL', 'NG.LOW')
0.0390 ('SIBN.VOL', 'SIBN.CLOSE')
0.0390 ('SIBN.CLOSE', 'SIBN.VOL')
0.0331 ('VTBR.CLOSE', 'NVTK.HIGH')
0.0331 ('NVTK.HIGH', 'VTBR.CLOSE')
0.0301 ('SIBN.VLT', 'BZ.VOLR')
0.0301 ('BZ.VOLR', 'SIBN.VLT')
0.0288 ('GMKN.AVG', 'LKOH.HIGH')
0.0288 ('LKOH.HIGH', 'GMKN.AVG')
0.0262 ('SIBN.LOW', 'VTBR.AVG')
0.0262 ('VTBR.AVG', 'SIBN.LOW')
0.0256 ('LKOH.OPEN', 'NVTK.VOLR')
0.0256 ('NVTK.VOLR', 'LKOH.OPEN')
0.0255 ('NG.OPEN', 'GMKN.LOW')
0.0255 ('GMKN.LOW', 'NG.OPEN')
0.0247 ('GAZP.OPEN', 'GAZP.VOL')
0.0247 ('GAZP.VOL', 'GAZP.OPEN')
0.0246 ('GMKN.CLOSE', 'LKOH.VLT')
0.0246 ('LKOH.VLT', 'GMKN.CLOSE')
mutual top10 mean:
0.0451 ('NG.LOW', 'VTBR.VOL')
0.0451 ('VTBR.VOL', 'NG.LOW')
0.0390 ('SIBN.VOL', 'SIBN.CLOSE')
0.0390 ('SIBN.CLOSE', 'SIBN.VOL')
0.0331 ('VTBR.CLOSE', 'NVTK.HIGH')
0.0331 ('NVTK.HIGH', 'VTBR.CLOSE')
0.0301 ('SIBN.VLT', 'BZ.VOLR')
0.0301 ('BZ.VOLR', 'SIBN.VLT')
0.0288 ('GMKN.AVG', 'LKOH.HIGH')
0.0288 ('LKOH.HIGH', 'GMKN.AVG')
0.0262 ('SIBN.LOW', 'VTBR.AVG')
0.0262 ('VTBR.AVG', 'SIBN.LOW')
0.0256 ('LKOH.OPEN', 'NVTK.VOLR')
0.0256 ('NVTK.VOLR', 'LKOH.OPEN')
0.0255 ('NG.OPEN', 'GMKN.LOW')
0.0255 ('GMKN.LOW', 'NG.OPEN')
0.0247 ('GAZP.OPEN', 'GAZP.VOL')
0.0247 ('GAZP.VOL', 'GAZP.OPEN')
0.0246 ('GMKN.CLOSE', 'LKOH.VLT')
0.0246 ('LKOH.VLT', 'GMKN.CLOSE')
'''    
    
    
'''
Только газпром
epoch: 300
    avg: 0.08875966711771097
    avg size: 128.77
    best: 0.089446580979145 (height:22 size:123)
    ss.pct
      x1=ss.pct
        x1=rolling.max
          x=s.square
            x=s.square
              x=GAZP.VLT
          window=30
        x2=sc.sum
          x=GAZP.LOW
          c=4915.8150301891
      x2=sc.sum
        x=ss.min
          x1=GAZP.LOW
          x2=sc.sum
            x=ss.pct
              x1=rolling.max
                x=rolling.max
                  x=s.square
                    x=s.square
                      x=s.square
                        x=GAZP.VLT
                  window=30
                window=20
              x2=sc.sum
                x=rolling.max
                  x=s.square
                    x=rolling.max
                      x=ss.min
                        x1=sc.sum
                          x=ss.pct
                            x1=rolling.max
                              x=rolling.max
                                x=s.square
                                  x=s.square
                                    x=s.square
                                      x=GAZP.VLT
                                window=30
                              window=19
                            x2=sc.sum
                              x=ss.pct
                                x1=rolling.max
                                  x=s.square
                                    x=s.square
                                      x=GAZP.VLT
                                  window=29
                                x2=sc.sum
                                  x=ss.sum
                                    x1=rolling.std
                                      x=GAZP.CLOSE
                                      window=2
                                    x2=GAZP.OPEN
                                  c=4915.8150301891
                              c=62700.792455666626
                          c=4138.172190564969
                        x2=ss.min
                          x1=rolling.max
                            x=ss.pct
                              x1=rolling.max
                                x=rolling.max
                                  x=rolling.max
                                    x=s.square
                                      x=s.square
                                        x=s.square
                                          x=GAZP.VLT
                                    window=30
                                  window=19
                                window=19
                              x2=ss.pct
                                x1=sc.sum
                                  x=ss.sum
                                    x1=rolling.std
                                      x=s.square
                                        x=GAZP.VLT
                                      window=2
                                    x2=GAZP.OPEN
                                  c=4138.172190564969
                                x2=sc.sum
                                  x=ss.pct
                                    x1=GAZP.OPEN
                                    x2=sc.sum
                                      x=ss.sum
                                        x1=GAZP.LOW
                                        x2=GAZP.OPEN
                                      c=4915.8150301891
                                  c=62700.792455666626
                            window=2
                          x2=sc.sum
                            x=ss.pct
                              x1=rolling.max
                                x=ss.pct
                                  x1=rolling.max
                                    x=rolling.max
                                      x=rolling.max
                                        x=s.square
                                          x=s.square
                                            x=s.square
                                              x=GAZP.VLT
                                        window=30
                                      window=19
                                    window=19
                                  x2=GAZP.VLT
                                window=2
                              x2=sc.sum
                                x=ss.pct
                                  x1=rolling.min
                                    x=s.square
                                      x=s.square
                                        x=GAZP.VLT
                                    window=21
                                  x2=sc.sum
                                    x=ss.sum
                                      x1=rolling.std
                                        x=GAZP.LOW
                                        window=2
                                      x2=GAZP.OPEN
                                    c=4915.8150301891
                                c=62700.792455666626
                            c=4915.8150301891
                      window=29
                  window=31
                c=9877.311700550852
            c=2209.4194890281965
        c=205867.92329767192

'''
    


'''
['1_GAZP.csv', '1_LKOH.csv', '1_ROSN.csv', '1_VTBR.csv', '1_GMKN.csv', '1_NVTK.csv', '1_SIBN.csv',
                     '24_NG.csv', '24_BZ.csv', '45_EUR_RUB__TOD.csv', '45_USD000000TOD.csv']
epoch: 946
    avg: 0.12255972410300041
    avg size: 436.2133333333333
    best: 0.12484087463125969 (height:21 size:395)
    ss.min
      x1=rolling.sum
        x=rolling.min
          x=sf.ewm
            x=sc.sum
              x=ss.min
                x1=rolling.min
                  x=rolling.min
                    x=rolling.max
                      x=rolling.min
                        x=BZ.VOLR
                        window=16
                      window=6
                    window=7
                  window=9
                x2=ss.min
                  x1=rolling.max
                    x=sp.shift
                      x=sf.ewm
                        x=ss.mul
                          x1=rolling.min
                            x=sp.shift
                              x=rolling.max
                                x=BZ.VOLR
                                window=3
                              periods=15
                            window=8
                          x2=ss.min
                            x1=sf.ewm
                              x=ss.mul
                                x1=ss.min
                                  x1=ss.min
                                    x1=sf.ewm
                                      x=rolling.min
                                        x=sc.sum
                                          x=VTBR.LOW
                                          c=2800326.7328868285
                                        window=8
                                      a=0.12375559461182895
                                    x2=rolling.max
                                      x=rolling.min
                                        x=BZ.VOLR
                                        window=16
                                      window=6
                                  x2=ss.lt
                                    x1=rolling.var
                                      x=sc.sum
                                        x=rolling.min
                                          x=BZ.VOLR
                                          window=12
                                        c=-0.12973607482032334
                                      window=13
                                    x2=rolling.min
                                      x=VTBR.VOLR
                                      window=16
                                x2=ss.min
                                  x1=BZ.VOLR
                                  x2=ss.lt
                                    x1=rolling.min
                                      x=sc.sum
                                        x=VTBR.CLOSE
                                        c=2800326.7328868285
                                      window=4
                                    x2=rolling.mean
                                      x=sc.sum
                                        x=VTBR.CLOSE
                                        c=2800326.7328868285
                                      window=15
                              a=0.6409745199664422
                            x2=ss.lt
                              x1=rolling.min
                                x=sc.sum
                                  x=VTBR.CLOSE
                                  c=2800326.7328868285
                                window=4
                              x2=rolling.max
                                x=rolling.min
                                  x=sc.sum
                                    x=VTBR.CLOSE
                                    c=2800326.7328868285
                                  window=4
                                window=16
                        a=0.801202183835528
                      periods=16
                    window=15
                  x2=ss.min
                    x1=sf.ewm
                      x=sc.sum
                        x=VTBR.HIGH
                        c=2800326.7328868285
                      a=0.14489400455033716
                    x2=rolling.max
                      x=rolling.min
                        x=BZ.VOLR
                        window=16
                      window=6
              c=2800326.7328868285
            a=0.5561561761281955
          window=9
        window=16
      x2=ss.min
        x1=ss.min
          x1=sf.ewm
            x=sc.sum
              x=ss.min
                x1=rolling.sum
                  x=rolling.min
                    x=sf.ewm
                      x=sc.sum
                        x=ss.min
                          x1=ss.mul
                            x1=rolling.min
                              x=sf.ewm
                                x=rolling.max
                                  x=rolling.min
                                    x=BZ.VOLR
                                    window=16
                                  window=6
                                a=0.55925781955219
                              window=10
                            x2=ss.min
                              x1=rolling.min
                                x=sc.sum
                                  x=VTBR.CLOSE
                                  c=2800326.7328868285
                                window=4
                              x2=rolling.max
                                x=rolling.min
                                  x=sc.sum
                                    x=VTBR.CLOSE
                                    c=2800326.7328868285
                                  window=4
                                window=15
                          x2=ss.min
                            x1=rolling.min
                              x=BZ.VOLR
                              window=16
                            x2=ss.min
                              x1=rolling.min
                                x=sf.ewm
                                  x=sc.sum
                                    x=VTBR.CLOSE
                                    c=2800326.7328868285
                                  a=0.3233712696861757
                                window=5
                              x2=rolling.max
                                x=rolling.min
                                  x=BZ.VOLR
                                  window=16
                                window=6
                        c=2800326.7328868285
                      a=0.5561561761281955
                    window=9
                  window=16
                x2=ss.min
                  x1=rolling.sum
                    x=VTBR.CLOSE
                    window=16
                  x2=ss.min
                    x1=rolling.sum
                      x=sf.ewm
                        x=sc.sum
                          x=ss.min
                            x1=rolling.max
                              x=rolling.min
                                x=rolling.min
                                  x=rolling.std
                                    x=rolling.max
                                      x=BZ.VOLR
                                      window=16
                                    window=14
                                  window=8
                                window=9
                              window=7
                            x2=rolling.std
                              x=sf.ewm
                                x=sc.sum
                                  x=VTBR.CLOSE
                                  c=2800326.7328868285
                                a=0.55925781955219
                              window=16
                          c=2880180.2913517314
                        a=0.6409745199664422
                      window=9
                    x2=ss.min
                      x1=rolling.min
                        x=sf.ewm
                          x=sc.sum
                            x=VTBR.CLOSE
                            c=1981668.0299277552
                          a=0.09951064121671954
                        window=9
                      x2=BZ.VOLR
              c=1391831.4381502764
            a=0.5212499328058415
          x2=rolling.min
            x=rolling.min
              x=rolling.std
                x=rolling.max
                  x=BZ.VOLR
                  window=6
                window=14
              window=8
            window=9
        x2=ss.min
          x1=rolling.sum
            x=rolling.min
              x=sf.ewm
                x=sc.sum
                  x=ss.div
                    x1=rolling.min
                      x=rolling.min
                        x=rolling.max
                          x=rolling.min
                            x=BZ.VOLR
                            window=16
                          window=9
                        window=8
                      window=9
                    x2=ss.min
                      x1=rolling.min
                        x=sc.sum
                          x=BZ.VOLR
                          c=2800326.7328868285
                        window=8
                      x2=ss.min
                        x1=sf.ewm
                          x=rolling.min
                            x=sc.sum
                              x=VTBR.CLOSE
                              c=2800326.7328868285
                            window=8
                          a=0.12375559461182895
                        x2=rolling.max
                          x=rolling.min
                            x=BZ.VOLR
                            window=16
                          window=6
                  c=2800326.7328868285
                a=0.5561561761281955
              window=8
            window=4
          x2=ss.min
            x1=rolling.min
              x=LKOH.VOL
              window=6
            x2=ss.min
              x1=rolling.min
                x=LKOH.VOL
                window=4
              x2=ss.min
                x1=rolling.sum
                  x=rolling.min
                    x=sf.ewm
                      x=sc.sum
                        x=ss.min
                          x1=rolling.min
                            x=sp.shift
                              x=sp.shift
                                x=BZ.VOLR
                                periods=16
                              periods=16
                            window=30
                          x2=ss.min
                            x1=rolling.min
                              x=BZ.VOLR
                              window=16
                            x2=ss.min
                              x1=rolling.min
                                x=sf.ewm
                                  x=sc.sum
                                    x=VTBR.CLOSE
                                    c=2800326.7328868285
                                  a=0.19881650398021178
                                window=5
                              x2=ss.min
                                x1=sf.ewm
                                  x=sc.sum
                                    x=VTBR.HIGH
                                    c=2800326.7328868285
                                  a=0.26446517568039846
                                x2=rolling.max
                                  x=rolling.min
                                    x=BZ.VOLR
                                    window=16
                                  window=6
                        c=2800326.7328868285
                      a=0.5561561761281955
                    window=9
                  window=16
                x2=ss.min
                  x1=rolling.sum
                    x=rolling.min
                      x=sf.ewm
                        x=sc.sum
                          x=ss.min
                            x1=rolling.max
                              x=rolling.min
                                x=BZ.VOLR
                                window=16
                              window=8
                            x2=ss.min
                              x1=sc.sum
                                x=ss.min
                                  x1=BZ.VOLR
                                  x2=ss.mul
                                    x1=sc.sum
                                      x=VTBR.CLOSE
                                      c=5831087.519119406
                                    x2=ss.min
                                      x1=rolling.min
                                        x=sc.sum
                                          x=VTBR.CLOSE
                                          c=2800326.7328868285
                                        window=4
                                      x2=rolling.max
                                        x=rolling.min
                                          x=sc.sum
                                            x=VTBR.CLOSE
                                            c=2800326.7328868285
                                          window=4
                                        window=15
                                c=0.5561561761281955
                              x2=ss.min
                                x1=sc.sum
                                  x=VTBR.CLOSE
                                  c=2800326.7328868285
                                x2=rolling.max
                                  x=rolling.min
                                    x=BZ.VOLR
                                    window=16
                                  window=15
                          c=2800326.7328868285
                        a=0.6409745199664422
                      window=9
                    window=16
                  x2=ss.min
                    x1=rolling.sum
                      x=sf.ewm
                        x=sc.sum
                          x=ss.min
                            x1=rolling.max
                              x=rolling.min
                                x=BZ.VOLR
                                window=16
                              window=7
                            x2=ss.min
                              x1=sc.sum
                                x=ss.min
                                  x1=sp.shift
                                    x=BZ.VOLR
                                    periods=5
                                  x2=ss.min
                                    x1=GAZP.VOLR
                                    x2=ss.lt
                                      x1=rolling.min
                                        x=sc.sum
                                          x=VTBR.CLOSE
                                          c=2800326.7328868285
                                        window=4
                                      x2=rolling.max
                                        x=rolling.min
                                          x=sc.sum
                                            x=VTBR.CLOSE
                                            c=2800326.7328868285
                                          window=4
                                        window=14
                                c=2800326.7328868285
                              x2=ss.min
                                x1=rolling.min
                                  x=sc.sum
                                    x=VTBR.CLOSE
                                    c=2913693.28313512
                                  window=8
                                x2=rolling.max
                                  x=rolling.min
                                    x=BZ.VOLR
                                    window=16
                                  window=16
                          c=2880180.2913517314
                        a=0.6409745199664422
                      window=9
                    x2=ss.min
                      x1=rolling.min
                        x=sf.ewm
                          x=sc.sum
                            x=VTBR.CLOSE
                            c=2800326.7328868285
                          a=0.55925781955219
                        window=9
                      x2=rolling.max
                        x=rolling.min
                          x=BZ.VOLR
                          window=16
                        window=6



'''

'''
газпром и сопутствующие ограничение размера дерева до 120
epoch: 1602
    finished at: 2020-03-14 12:24:49.964017   elapsed: 0:00:20.419167
    avg: 0.05785171442154114
    avg size: 92.78
    last inc score at: 1504
    best: 0.09179499040414311 (height:14 size:99)


    ss.sum
      x1=ss.sum
        x1=ss.sum
          x1=ss.sum
            x1=ss.sum
              x1=VTBR.AVG
              x2=VTBR.CLOSE
            x2=ss.xor
              x1=sc.gt
                x=ss.sum
                  x1=SIBN.CLOSE
                  x2=BZ.VOLR
                c=999986.7344603252
              x2=ss.lt
                x1=NG.LOW
                x2=GMKN.VLT
          x2=ss.sum
            x1=ss.sum
              x1=ss.sum
                x1=VTBR.HIGH
                x2=ss.sum
                  x1=rolling.max
                    x=GAZP.VOL
                    window=13
                  x2=rolling.std
                    x=sc.gt
                      x=ss.sum
                        x1=BZ.VOLR
                        x2=GAZP.VOL
                      c=723669.9362119874
                    window=6
              x2=sc.sum
                x=ss.xor
                  x1=ss.sum
                    x1=BZ.VOLR
                    x2=LKOH.OPEN
                  x2=sp.shift
                    x=sc.sum
                      x=ss.pct
                        x1=LKOH.OPEN
                        x2=ss.sum
                          x1=GAZP.VOL
                          x2=sc.mul
                            x=ss.lt
                              x1=NG.LOW
                              x2=GMKN.VLT
                            c=-28690.401637805506
                      c=38079.206067580024
                    periods=11
                c=-15435.197996019233
            x2=ss.sum
              x1=ss.sum
                x1=rolling.max
                  x=GAZP.VOL
                  window=13
                x2=rolling.std
                  x=sc.gt
                    x=ss.sum
                      x1=SIBN.CLOSE
                      x2=BZ.VOLR
                    c=723669.9362119874
                  window=7
              x2=rolling.sum
                x=VTBR.AVG
                window=11
        x2=ss.sum
          x1=sc.sum
            x=ss.xor
              x1=rolling.std
                x=ss.lt
                  x1=NG.LOW
                  x2=SIBN.CLOSE
                window=6
              x2=sp.shift
                x=sc.sum
                  x=ss.pct
                    x1=LKOH.OPEN
                    x2=ss.sum
                      x1=GAZP.VOL
                      x2=sc.sum
                        x=rolling.max
                          x=GAZP.VOL
                          window=11
                        c=723669.9362119874
                  c=38079.206067580024
                periods=11
            c=118581.11310185694
          x2=rolling.sum
            x=sc.gt
              x=ss.sum
                x1=BZ.VOLR
                x2=LKOH.OPEN
              c=160335.1386053733
            window=11
      x2=ss.xor
        x1=NG.LOW
        x2=ss.sum
          x1=BZ.VOLR
          x2=VTBR.CLOSE
'''    
    
