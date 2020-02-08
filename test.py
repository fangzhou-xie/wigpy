import matplotlib.pyplot as plt

from utils import readdata
from wig import WIG


def test(ct=0):
    data = readdata()
    wig = WIG(data, compress_topk=ct, epochs=1, min_count=1)
    wig.train()
    df = wig.generateindex(compare=True)
    testplot(df)


def testplot(df):
    df.plot(y=['index', 'indexaag', 'indexori', 'indexwig'])
    plt.show()


def main():
    for ct in [0, 1000]:
        test(ct)
    # testplot()
    pass


if __name__ == "__main__":
    main()
