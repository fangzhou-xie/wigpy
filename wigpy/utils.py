import os
import sqlite3 as lite
import time
from datetime import datetime

module_path = os.path.dirname(__file__)
datafile = 'headlines.db'


def timer(method):
    """timer decorator"""
    def timed(*args, **kw):
        starttime = time.time()
        result = method(*args, **kw)
        endtime = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((endtime - starttime) * 1000)
        else:
            deltatime = endtime - starttime
            if deltatime < 1:
                print('{} {} time : {:2.5f} ms'.format(datetime.now(), method.__name__,
                                                       (endtime - starttime) * 1000))
            elif deltatime > 60:
                print('{} {} time : {:2.5f} min'.format(datetime.now(), method.__name__,
                                                        (endtime - starttime) / 60))
            else:
                print('{} {} time : {:2.5f} s'.format(datetime.now(), method.__name__,
                                                      (endtime - starttime)))
        return result
    return timed


def readdata(summary=False):
    conn = lite.connect(os.path.join(module_path, datafile))
    c = conn.cursor()
    if summary:
        result = c.execute(
            'select date, title, summary from headlines;').fetchall()
    else:
        result = c.execute(
            'select date, title from headlines;').fetchall()
    conn.close()
    return result
