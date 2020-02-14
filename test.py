import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
import seaborn as sns

from utils import readdata
from wig import WIG

sns.set_style("darkgrid")
# pd.options.mode.chained_assignment = None
years_fmt = mdates.DateFormatter('%Y')


def test(ct=0):
    data = readdata()
    wig = WIG(data, prune_topk=ct, epochs=1, min_count=1)
    # wig.train()
    df = wig.generateindex(compare=True)
    testplot(df, ct)


def testplot(df, ct):
    p = df.plot(y=['index', 'indexwig', 'indexaag', 'indexori'])
    p.xaxis.set_major_locator(mdates.YearLocator())  #
    p.xaxis.set_major_formatter(years_fmt)
    # p.xaxis.set_label_coords(0.01, 0)
    p.set_xlabel('Month')
    p.set_ylabel('Index')
    p.set_title('Economic Policy Uncertainty Index (EPU)')
    p.legend(['EPU_COM', 'EPU_WIG', 'EPU_LDA', 'EPU'], loc='upper left')
    fig = p.get_figure()
    fig.savefig('results/test_{}.png'.format(ct), dpi=300, bbox_inches='tight')


def main():
    # for ct in [0, 1000]:
    #     test(ct)
    test(1000)
    # testplot()
    pass


if __name__ == "__main__":
    main()
