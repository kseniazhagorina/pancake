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
import target
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
    return (x - minx)/diff 
MINMAX = rolling('minmax', lambda x, window: minmax(x, window))

            
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
DROPNA = series_unary_op('dropna', lambda x: x.dropna())
REIND = series_unary_op('reind', lambda x: x.reindex(pd.date_range(x.index[0], x.index[-1])))


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
                  SNOT, SABS, SQRT, SQR, SLOG, CUMMIN, CUMMAX, NSUM, NMUL, NLT, NGT, # CUMSUM,
                  EWM, EWM2, SHIFT, CHANGE, FFILLNA, FILLNA0, MINMAX, DROPNA #, REIND,  
                  ]


                           
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
    def __init__(self, operations, series, target, n, p_series=None, p_operations=None,  max_node_size=None, max_n=None, best_n=None):    
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
        self.__p_operations = p_operations or {}
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
            info = mutual_info_classif(v, self.__target, n_neighbors=15, random_state=4838474)[0]
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

GROWTH = lambda thr: lambda d: target.growth(d, lambda s: target.split_thr(s, thr=thr))
CHANGE = lambda thr: lambda d: target.change(d, lambda s: target.split_thr(s, thr=thr))   

def run(start, end, target_params, series_filenames, n, max_epoch, save_as, p_series=None, expand=False): 
    print(target_params)
    t = target.load_target(*target_params).TARGET[start:end]
    series = target.load_series(series_filenames, expand)
    ps = None
    if p_series:
        ps = np.array([p_series[s.name] if s.name in p_series else min(p_series.values()) for s in series])
        ps = ps - ps.min()
        ps = list(0.5*ps/ps.sum() + 0.5/len(ps))

    
    
    # при N = max_node_size * max_n затраты памяти будут примерно N*100 Кб
    g = GenProg(ALL_OPERATIONS, series, t, n, p_series=ps, max_n=n*2, max_node_size=120)
    g.start()
    g.print_state()
    while max_epoch is None or g.epoch < max_epoch:
        g.next_epoch()
        g.print_state()
        g.best.value.to_csv(save_as)
        if g.epoch % 20 == 0:
            gc.collect()
        
   
    
def calcstat(start, end, target_params, series_filenames, output_filename, expand=False):
    t = target.load_target(*target_params)[start:end]
    s = target.load_series(series_filenames, expand)
    g = GenProg(ALL_OPERATIONS, s, t, 100)
    
    from collections import defaultdict
    individual = defaultdict(list)
    mutual = defaultdict(list)
    operations = defaultdict(list)
    
    def get_child_series(node):
        if isinstance(node, LeafNode) and node.value_type == ValueType.SERIES:
            return [node.value.name]
        result = []
        for c in node.children:
            result += get_child_series(c)
        return result
    
    def get_operations(node):
        if isinstance(node, LeafNode):
            return []
        result = [node.func.name]
        for c in node.children:
            result += get_operations(c)
        return result
        

    def save_result():
        def order_by_mean_top_10p(scores):
            return list(
                sorted([
                   (s, np.mean(list(sorted(v, reverse=True))[0:int(max(len(v)/10, 10))]), len(v))
                   for s, v in scores.items()
                   if len(v) >= 10
                ], key=lambda x: x[1], reverse=True))
                
        result = {
            'individual': order_by_mean_top_10p(individual),
            'mutual': order_by_mean_top_10p(mutual),
            'operations': order_by_mean_top_10p(operations)
            }
        
        with codecs.open(output_filename, 'w', 'utf-8') as out:
            out.write(json.dumps(result, ensure_ascii=False, indent=4))
    
        
    uniq = set()    
        
    for i in range(300000):
        if i%100 == 0:
            print(i)
        if i%1000 == 0:
            save_result()
        node = g.generate(h=3)
        s = print_node(node)
        if s in uniq:
            continue
        uniq.add(s)
        children = get_child_series(node)
        ops = get_operations(node)
        
        uniq_children = list(sorted(set(children)))
        uniq_ops = list(sorted(set(ops)))
        v = g.eval(node)
        if v == -np.inf:
            continue
        for c in uniq_children:
            individual[c].append(v)
            for c1 in uniq_children:
                if c != c1 or children.count(c) > 1:
                    mutual[(c,c1)].append(v)
                if c == c1:
                    break
        for o in uniq_ops:
            operations[o].append(v)
            
    save_result()
            
    

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
    
P_SERIES_NEW = dict(((key, prob) for key, prob, count in json.loads(codecs.open('probabilities_GAZP.json', 'r', 'utf-8').read())['individual']))
P_SERIES_SELF = dict(((key, prob) for key, prob, count in json.loads(codecs.open('probabilities_GAZP_self.json', 'r', 'utf-8').read())['individual']))

    
def main1():
    run('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
        ['1_GAZP.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_best_new_target.csv')
        
def main11():
    run('2009-01-01',
        '2018-12-31',
        ['1_GAZP.csv', CHANGE(0.002)], 
        ['1_GAZP.csv'],
        n=100,
        max_epoch=None,
        expand=True,
        save_as='gazp_close_close_002.csv')        
        
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
        ['1_GAZP.csv', GROWTH(0.01)], 
        ['1_LKOH.csv', '1_ROSN.csv', '1_NVTK.csv', '1_SIBN.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_similar_close_high_01.csv')
        
def main4():
    run('2009-01-01',
        '2018-12-31',
        ['1_GAZP.csv', GROWTH(0.015)], 
       ['1_AKRN.csv', '1_NVTK.csv', '1_NKNC.csv', '1_VSMO.csv',  '1_TRNFP.csv', '1_GMKN.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_top6_close_high_015.csv',
        p_series=P_SERIES_NEW)
        
def main5():
    run('2009-01-01',
        '2018-12-31',
        ['1_GAZP.csv', GROWTH(0.02)], 
        ['91_IMOEX.csv', '91_MOEX10.csv', '91_MOEXOG.csv', '91_MOEXFN.csv', '1_GAZP.csv'],
        n=100,
        max_epoch=None,
        save_as='gazp_imoex_close_high_02.csv')       

        
def main_calcstat():
    calcstat('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
        os.listdir('finam/data'),
        'probabilities_GAZP.json')
        
def main_calcstat1():
    calcstat('2009-01-01',
        '2018-12-31',
        '1_GAZP.csv', 
        ['1_GAZP.csv'],
        'probabilities_GAZP_self.json',
        expand=True)       

    
    
if __name__ == '__main__':
    #instruments = json.loads(codecs.open('finam/instruments/instruments.json', 'r', 'utf-8').read())
    #markets = json.loads(codecs.open('finam/markets.json', 'r', 'utf-8').read())

    main3()

