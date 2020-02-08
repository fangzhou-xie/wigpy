import sqlite3 as lite

from wig import WIG

datafile = 'headlines.sqlite'


def read():
    conn = lite.connect(datafile)
    c = conn.cursor()
    result = c.execute('select date, title from headlines_reduce;').fetchall()
    conn.close()
    return result


def main():
    data = read()
    wig = WIG(data, compress_topk=0)
    wig.train(loss_per_batch=True)
    pass


if __name__ == "__main__":
    main()
