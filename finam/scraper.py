#!usr/bin/env
# -*- coding: utf-8 -*-

import requests
from datetime import timedelta, date, datetime
import urllib
import os
import shutil
import time
import sys
import json
import codecs



def download(instrument, date_from, date_to, output_file):
    
    params = [  
        ('market', instrument['market']),
        ('em', instrument['id']),
        ('code', instrument['code']),
        ('apply', '0'),
        ('df', date_from.day), # df, mf, yf, from, dt, mt, yt, to – это параметры времени.
        ('mf', date_from.month - 1),
        ('yf',date_from.year),
        ('from', date_from.strftime('%d.%m.%Y')),
        ('dt', date_to.day),
        ('mt', date_to.month - 1),
        ('yt', date_to.year),
        ('to', date_from.strftime('%d.%m.%Y')),
        ('p', '2'), # p — период котировок (тики, 1 мин., 5 мин., 10 мин., 15 мин., 30 мин., 1 час, 1 день, 1 неделя, 1 месяц)
        ('f', instrument['code']),
        ('e', '.csv'), # e – расширение получаемого файла; возможны варианты — .txt либо .csv
        ('cn', instrument['code']),
        ('dtf', '1'), # dtf — формат даты (1 — ггггммдд, 2 — ггммдд, 3 — ддммгг, 4 — дд/мм/гг, 5 — мм/дд/гг)
        ('tmf', '1'), # tmf — формат времени (1 — ччммсс, 2 — ччмм, 3 — чч: мм: сс, 4 — чч: мм)
        ('MSOR', '0'), # MSOR — выдавать время (0 — начала свечи, 1 — окончания свечи)
        ('mstime', 'on'),
        ('mstimever', '1'), # mstimever — выдавать время (НЕ московское — mstimever=0; московское — mstime='on', mstimever='1')
        ('sep', '1'), # sep — параметр разделитель полей (1 — запятая (,), 2 — точка (.), 3 — точка с запятой (;), 4 — табуляция (»), 5 — пробел ( ))
        ('sep2', '1'), # sep2 — параметр разделитель разрядов (1 — нет, 2 — точка (.), 3 — запятая (,), 4 — пробел ( ), 5 — кавычка ('))
        ('datf', '5'), # datf — Перечень получаемых данных (#1 — TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL; #2 — TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE; #3 — TICKER, PER, DATE, TIME, CLOSE, VOL; #4  TICKER, PER, DATE, TIME, CLOSE; #5 — DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL; #6 — DATE, TIME, LAST, VOL, ID, OPER).
        ('at', '0'), # at — добавлять заголовок в файл (0 — нет, 1 — да)
        ('fsp', '1') # fsp- заполнять периоды без сделок
    ]
    
    path = '{}_{}_{}.csv'.format(instrument['code'], date_from.strftime("%Y%m%d"), date_to.strftime("%Y%m%d"))
    url = "http://export.finam.ru/" + path + "?" + urllib.parse.urlencode(params)
    headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/77.0.3865.90 Chrome/77.0.3865.90 Safari/537.36',
                'Cookie': 'ASPSESSIONIDSSTTATBR=IJCILAJCBFKAPCEIJEAGGIHP',
                'Host': 'export.finam.ru'}
    # http://export.finam.ru/GAZP_200123_200124.csv?market=1&em=16842&code=GAZP&apply=0&df=23&mf=0&yf=2020&from=23.01.2020&dt=24&mt=0&yt=2020&to=24.01.2020&p=2&f=GAZP_200123_200124&e=.csv&cn=GAZP&dtf=1&tmf=1&MSOR=0&mstime=on&mstimever=1&sep=1&sep2=1&datf=5&fsp=1
    print(url)
    response = requests.get(url, headers=headers)
    with open(output_file, 'ab') as out:
        out.write(response.content)

def last_downloaded_date(filename):
    if not os.path.exists(filename):
        return None
    lines = open(filename, 'r').readlines()
    if len(lines) == 0:
        return None 
    last = lines[-1].split(',')
    if len(last) == 0 or len(last[0]) == 0:
        return None
    return datetime.strptime(last[0], '%Y%m%d').date() 
    
def download_all(instruments, date_from, date_to, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for instrument in instruments:
        filename = os.path.join(output_dir, '{}_{}.csv'.format(instrument['market'], instrument['code']))
        if os.path.exists(filename):
            last = last_downloaded_date(filename)
            date_from = last + timedelta(days=1)
            
        f  = date_from
        while f <= date_to:
            t = f + timedelta(days=365)
            if t > date_to:
                t = date_to
            print('[{}] Load {} from {} to {}'.format(datetime.now(), instrument['name'], f.strftime('%Y.%m.%d'), t.strftime('%Y.%m.%d')))
            try:
                download(instrument, f, t, filename)
                print('[{}] Success'.format(datetime.now()))
                time.sleep(5)
            except Exception as e:
                print('[{}] Error {}'.format(datetime.now(), e))
                os.remove(filename)
                break
            f = t + timedelta(days=1)
            
if __name__ == '__main__':
    instruments = json.loads(codecs.open(sys.argv[1], 'r', 'utf-8').read())
    date_from = datetime.strptime(sys.argv[2], '%d.%m.%Y').date()
    date_to = datetime.strptime(sys.argv[3], '%d.%m.%Y').date()
    output_dir = sys.argv[4]
    download_all(instruments, date_from, date_to, output_dir)
    
    
    

