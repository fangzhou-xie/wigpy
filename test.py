import sqlite3 as lite

from wig import WIG

datafile = 'headlines_reduce.tsv'


def read():
    conn = lite.connect(datafile)
    c = conn.cursor()
    result = c.execute('select date, title from headliens_reduce;').fetchall()
    conn.close()
    return result


def main():
    data = read()
    wig = WIG()
    pass


if __name__ == "__main__":
    main()
