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
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import minimize
import statsmodels.api as sm

import information_gain as ig
from correlation import time_series_rolling_corr

np.random.seed(838373)

def random_choice(x, p=None):
    if p is None:
        return x[np.random.randint(low=0, high=len(x))]
    else:
        s = sum(p)
        p = [v/s for v in p] 
        ids = [i for i in range(len(x))]
        return x[np.random.choice(ids, p=p)]

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
SDIV = series_op('div', lambda x1, x2: x1 / x2) # x2/x2 = 1 или NaN чтобы не возникало деления на 0
SPCNT = series_op('pct', lambda x1, x2: (x1 - x2) / x1)
SMAX = series_op('max', lambda x1, x2: (x1.gt(x2)).astype(int)*x1 + (x1.lt(x2)).astype(int)*x2)
SMIN = series_op('min', lambda x1, x2: (x1.gt(x2)).astype(int)*x2 + (x1.lt(x2)).astype(int)*x1)
SLT = series_op('lt', lambda x1, x2: (x1.lt(x2)).astype(int))
SAND = series_op('and', lambda x1, x2: (((x1 > 0).astype(int) + (x2 > 0).astype(int)) > 1).astype(int))
SOR = series_op('or', lambda x1, x2: (((x1 > 0).astype(int) + (x2 > 0).astype(int)) > 0).astype(int))
SXOR = series_op('xor', lambda x1, x2: (((x1 > 0).astype(int) + (x2 > 0).astype(int)) == 1).astype(int))

def series_unary_op(name, calc):
    return  Func('s.{}'.format(name),
            arguments=[Argument('x', ValueType.SERIES)],
            result_type=ValueType.SERIES,
            calc = calc)

SNOT = series_unary_op('not', lambda x: (x <= 0).astype(int))
SABS = series_unary_op('abs', lambda x: x.abs())
SQRT = series_unary_op('sqrt', lambda x: x.apply(lambda v: math.sqrt(v) if v >= 0 else np.nan))
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
                       Argument('c', ValueType.INT, interval=pd.Interval(left=-5, right=5, closed='both'))],
            result_type=ValueType.SERIES,
            calc = calc)

NSUM = series_num_op('sum', lambda x, c: x + c)
NMUL = series_num_op('mul', lambda x, c: x * c)
NLT = series_num_op('lt', lambda x, c: (x < c).astype(int))
NGT = series_num_op('gt', lambda x, c: (x > c).astype(int))

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


ALL_OPERATIONS = [ROLLING_MEAN, ROLLING_MIN, ROLLING_MAX, ROLLING_STD, ROLLING_VAR, ROLLING_SUM, ROLLING_CORR,
                  SSUM, SSUB, SMUL, SDIV, SPCNT, SMAX, SMIN, SLT, SAND, SOR, SXOR,
                  SNOT, SABS, SQRT, SQR, SLOG, CUMMIN, CUMMAX, NSUM, NMUL, NLT, NGT, # CUMSUM
                  EWM, EWM2, SHIFT, CHANGE, FFILLNA, FILLNA0]


                           
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

        values = [c.value for c in self.__children]
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
    def __init__(self, operations, series, target, n):    
        self.__operations = operations
        self.__series = [LeafNode(ValueType.SERIES, s) for s in series]
        self.__target = target
        self.__count = n
        self.__items = []
        self.epoch = 0
        self.__last_epoch_inc_score = 0
    
    @property
    def best(self):
        return self.__items[0][0]
    
    def start(self):
        self.__items = self.eval_all([self.generate() for i in range(self.__count)])
        
    def print_state(self):
        print('epoch: {}'.format(self.epoch))
        best, score = self.__items[0]
        if self.__last_epoch_inc_score == self.epoch:
            print(print_node(best, intent='    '))
        print('\n')
        print('epoch: {}'.format(self.epoch))
        print('    avg: {}'.format(np.mean([x[1] for x in self.__items if x[1] != -np.inf])))
        print('    avg size: {}'.format(np.mean([x[0].size for x in self.__items])))
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
            return random_choice(self.__series)
        return None
        
    
    def mutate_leaf(self, node, argument):
        if node.value_type == ValueType.SERIES:
            return random_choice(self.__series)
    
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
            return random_choice(self.__series)
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
        if node.value is None:
            return -np.inf
        # бесполезное прибавление числа и умножение на число в корне дерева ничего не меняет
        if isinstance(node, FuncNode) and node.func.name in [NSUM.name, NMUL.name, SABS.name]:
            return -np.inf
        
        v = node.value
        v = v + v/np.inf  # v/np.inf = 0 или inf/inf = NaN        
        v, t = node.value.align(self.__target, join='right', fill_value = 0)
               
        try:
            v = v.values.reshape(-1, 1)
            info = mutual_info_classif(v, self.__target, n_neighbors=5, random_state=4838474)[0]
            # return info - 0.0002 * node.height # - 0.000002 * node.size
            return info - 0.00001 * node.size
        except Exception as e:
            sys.stderr.write('{}'.format(e))
            return -np.inf      

    def eval_all(self, items):
        r = [(item, self.eval(item)) for item in items]
        return list(sorted(r, key=lambda x: x[1], reverse=True))
                 
    def next_epoch(self):
        '''3 лучших переходят в следующую эпоху как есть'''
        
        prev_best_score = self.__items[0][1]
        next = [x for x, v in self.__items[0:3]]
        nexthash = set(print_node(x) for x in next)
        
        p = [v for x, v in self.__items]
        m = min([v for v in p if v != -np.inf])
        p = [max(v - m, 0) for v in p]
        p = [v*v for v in p]

        
        while len(next) < 3*self.__count:
            rnd = np.random.random()
            x, _ = random_choice(self.__items, p=p)
            y, _ = random_choice(self.__items)
            new = self.cross(x, y) if rnd < 0.5 else self.cross(y, x)
            for i in range(random_choice([1,2,3], p=[0.5, 0.25, 0.1])):
                if new:
                    new = self.mutate(new)
            if new is not None and new.value is not None:
                hash = print_node(new)
                if hash not in nexthash:
                    next.append(new)
                    nexthash.add(hash)
        
        self.epoch += 1
        # 25% результата займут лучшие 75% - просто случайные
        evaluated = self.eval_all(next)
        n_best = self.__count//4
        top = evaluated[0:n_best]
        rest = evaluated[n_best:]
        np.random.shuffle(rest)     
        self.__items = top + rest[0:self.__count-n_best]
 
        if self.__items[0][1] != prev_best_score:
            self.__last_epoch_inc_score = self.epoch
        elif self.epoch - self.__last_epoch_inc_score > 10:
            c, v = self.select_best_child(self.__items[0][0])        
            if v > prev_best_score:
                print('We crop best to its best child')
                self.__items.push((c, v))
                self.__last_epoch_inc_score = self.epoch
        
def load_target(filename):
    d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
    growth = (d.HIGH - d.OPEN)/d.OPEN
    target = growth.apply(lambda x: int(x > 0.008)).shift(-1).dropna() # цель - рост на 0.8% в день
    return target
    
def load_series(filenames):
    series = []
    for filename in filenames:
        print('load {}'.format(filename))
        d = fnm.resample(fnm.read(os.path.join('finam/data', filename)), period='D')
        series += [d[c].rename(d.name+'.'+d[c].name) for c in d.columns]    
    return series
        
def run(start, end, target_filename, series_filenames, n, max_epoch, save_as): 
 
    target = load_target(target_filename)[start:end]
    series = load_series(series_filenames)
    
    g = GenProg(ALL_OPERATIONS, series, target, n)
    g.start()
    g.print_state()
    while(g.epoch < max_epoch):
        g.next_epoch()
        g.print_state()
        g.best.value.to_csv(save_as)
        
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
    

    
def main1():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_GAZP.csv'],
        n=100,
        max_epoch=300,
        save_as='gazp_best.csv')
        
def main2():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
       ['1_GAZP.csv', '1_LKOH.csv', '1_ROSN.csv',
        '1_VTBR.csv', '1_GMKN.csv', '1_NVTK.csv',
        '1_SIBN.csv', '24_NG.csv', '24_BZ.csv'],
        n=150,
        max_epoch=500,
        save_as='gazp_other_best.csv')
        
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

    main_calcstat()
    
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
    
    
