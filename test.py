import sqlite3 as lite

from wig import WIG

datafile = 'headlines.db'


def read():
    conn = lite.connect(datafile)
    c = conn.cursor()
    result = c.execute('select date, title from headlines_reduce;').fetchall()
    conn.close()
    return result


def crossvalidate():
    for emsize in [50, 100]:
        for batch_size in [32, 64]:
            for num_topics in [4, 10, 20]:
                for epochs in [3, 5]:
                    for lr in [0.001, 0.005]:
                        for compress_topk in [0, 1000]:
                            yield emsize, batch_size, num_topics, epochs, lr,\
                                compress_topk


def parasqlite(arg=None, query=True):
    conn = lite.connect('parameters.db')
    c = conn.cursor()
    c.execute('''create table if not exists parameters
                (emsize int, batch_size int, num_topics int, epochs int,
                lr real, compress_topk int);''')
    if query:
        c.execute('''insert or ignore into parameters
                    (emsize, batch_size, num_topics, epochs, lr, compress_topk, loss)
                    values (?,?,?,?,?,?,?);''', arg)
        conn.commit()
        conn.close()
    else:
        res = c.execute(
            'select * from parameters order by loss desc;').fetchone()
        conn.close()
        return res


def cv():
    data = read()
    for emsize, batch_size, num_topics, epochs, lr, compress_topk in crossvalidate():
        wig = WIG(data, emsize=emsize, batch_size=batch_size,
                  num_topics=num_topics, epochs=epochs, lr=lr,
                  compress_topk=compress_topk)
        loss = wig.train(False)
        arg = (emsize, batch_size, num_topics, epochs, lr, compress_topk, loss)
        parasqlite(arg, False)
    emsize, batch_size, num_topics, epochs, lr, compress_topk = parasqlite()
    print('best parameters are: {}'.format(
        emsize, batch_size, num_topics, epochs, lr, compress_topk))
    # TODO: last training or load existing models
    wig = WIG(data, emsize=emsize, batch_size=batch_size,
              num_topics=num_topics, epochs=epochs, lr=lr,
              compress_topk=compress_topk)
    wig.generateindex()


def main():
    data = read()
    wig = WIG(data, compress_topk=0, epochs=1, min_count=1)
    wig.train()
    wig.generateindex()
    pass


if __name__ == "__main__":
    main()
