import sqlite3 as lite

from wig import WIG

datafile = 'headlines.sqlite'


def read():
    conn = lite.connect(datafile)
    c = conn.cursor()
    result = c.execute('select date, title from headlines_reduce;').fetchall()
    conn.close()
    return result


def crossvalidate():
    for emsize in [10, 50, 100]:
        for batch_size in [32, 64]:
            for num_topics in [4, 10, 20]:
                for epochs in [3, 5]:
                    for lr in [0.001, 0.005]:
                        for compress_topk in [500, 1000]:
                            yield emsize, batch_size, num_topics, epochs, lr


def main():
    data = read()
    wig = WIG(data, compress_topk=0, epochs=1)
    wig.train(loss_per_batch=True)
    wig.generateindex()
    pass


if __name__ == "__main__":
    main()
