# coding: utf-8
import html
import math
import os
import pdb
import pickle
import re
import sqlite3 as lite
import sys
import time
import traceback
from calendar import monthrange as mr
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial

# from torch.utils.data._utils.collate import default_collate
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy  # with GPU support
import statsmodels.api as sm
import torch
import torch.multiprocessing as mp
# from colorama import init
from gensim.models import Word2Vec  # TODO: pytorch implemenation
# import nltk
from scipy.stats import entropy, pearsonr, spearmanr
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import scale
# from gensim.models import Phrases
# from gensim.models.phrases import Phraser
# from spacy.lang.en import LEMMA_EXC, LEMMA_INDEX, LEMMA_RULES
# from spacy.lemmatizer import Lemmatizer
# from nltk.stem.wordnet import WordNetLemmatizer
# from ot import dist
from torch import Tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset, random_split
from unidecode import unidecode

mp.set_sharing_strategy('file_system')
sns.set_style("darkgrid")
pd.options.mode.chained_assignment = None
years_fmt = mdates.DateFormatter('%Y')
torch.manual_seed(1)
# init()
# detect CUDA if possible, o.w. use CPU
if torch.cuda.is_available():
    spacy.require_gpu()
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
savefolder = './savemodel/'


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


class WigDataset(Dataset):
    def __init__(self, datalist):
        self.samples = datalist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class WIG():
    def __init__(self, datafile, tablename, paratb,
                 mode='all', dtype=torch.float32, period='M', summary=True):
        """
        data: .sqlite file
        tablename: name of the table

        self.start: start year in file
        self.end: end year in file
        self.mode: "all" or "year", refers to training all together or year by year
        self.dtype: torch.dtype, default: torch.float32
        self.period: 'Y', 'M', 'D', default: 'M'
        """
        assert datafile.endswith('.sqlite')
        self.data = datafile
        self.tb = tablename
        self.paratb = paratb
        self.dtype = dtype
        self.period = period
        self.summary = summary
        self.mode = mode
        conn = lite.connect(self.data)
        c = conn.cursor()
        self.start = int(c.execute("""SELECT strftime('%Y', date)
                    AS "Year" FROM {} ORDER BY Year ASC;""".format(self.tb)).fetchone()[0])
        self.end = int(c.execute("""SELECT strftime('%Y', date)
                    AS "Year" FROM {} ORDER BY Year DESC;""".format(self.tb)).fetchone()[0])
        conn.close()

    def __repr__(self):
        """print class name"""
        return f"Wasserstein Index Generation model, \
        with database file {self.data} \nand table name {self.tb}.\n\
        In addition, datatype: {self.dtype} and period :{self.period}."

    def parameter(self, paras):
        """
        hold a dictionary which contains current parameter set

        embd: embedding depth required by Word2Vec
        reg: regularization weight of Sinkhorn algorithm
        batch: training batch size
        topic: dimension of WIG you would like to reduce to
        rho: learning rate of Adam or SGD

        self.para: python dict
        """
        if len(paras) == 6:
            embd, reg, batch, topic, rho, loss = paras
        elif len(paras) == 5:
            embd, reg, batch, topic, rho = paras
        self.para = dict(embd=embd, reg=reg, batch=batch, topic=topic, rho=rho)
        print(f'this set of parameter is: {self.para}')

    def readdatayear(self, year):
        """read data from sql file"""
        ymlist = timeparse(year, self.period)
        ym_dftdict = defaultdict(list)
        # pdb.set_trace()
        # each = readsql(self.data, self.tb, year)
        for each in readsql(self.data, self.tb, year):
            if self.period == 'M':
                ymd = str(datetime.strptime(each[0], '%Y-%m-%d').date())[0:7]
            elif self.period == 'Y':
                ymd = str(datetime.strptime(each[0], '%Y-%m-%d').date())[0:4]
            elif self.period == 'D':
                ymd = str(datetime.strptime(each[0], '%Y-%m-%d').date())
            if ymd in ymlist:
                # TODO: add summary here
                if self.summary:
                    r = ' '.join(
                        [strreplace(each[1].replace(';', ' ')),
                         strreplace(each[3])])
                else:
                    r = strreplace(each[1].replace(';', ' '))
                ym_dftdict[ymd].append(r)
                # pdb.set_trace()
        return ym_dftdict

    def readdata(self):
        """get all data from sqlite file"""
        alldict = dict()
        for year in range(self.start, self.end + 1):
            alldict.update(self.readdatayear(year))
        self.titles = alldict

    def run(self, force_train=False):
        """this function controls behavior of running mode"""
        # pdb.set_trace()
        if self.mode == 'all':
            self.runall(force_train)
        elif self.mode == 'year':
            self.runyear(force_train)

    def runall(self, force_train=False):
        """the main model with cross validation"""
        self.readdata()
        writeallparas(self.paratb)
        result = sqlquery(self.paratb, cv=None, query=False)
        # pdb.set_trace()
        if len(result) > 0:
            for para in result:
                print('\n{}'.format(datetime.now()))
                # pdb.set_trace()
                self.parameter(para)  # change the state of self.para
                # embd, reg, batch, topic, rho, loss = self.para
                # self.prepare(worker=12)
                loss = indexgenerate(self.titles, self.para['embd'],
                                     self.para['reg'], self.para['batch'],
                                     self.para['topic'], self.para['rho'],
                                     test=True).detach()
                if loss.device == torch.device('cpu'):
                    err = float(loss)
                else:
                    err = float(loss.cpu())
                self.para.update({'loss': err})
                sqlquery(self.paratb, cv=self.para, query=False)
            print('running done for current parameters')
            embd, reg, batch, topic, rho = sqlquery(self.paratb)
            print('embd, reg, batch, topic, rho are chosen optimally as:',
                  embd, reg, batch, topic, rho)
            self.basis, self.lbd = indexgenerate(self.titles, embd, reg,
                                                 batch, topic, rho, test=False)
            self.save()
        elif len(result) > 1 & force_train:
            # self.prepare(worker=12)
            embd, reg, batch, topic, rho = sqlquery(self.paratb)
            self.basis, self.lbd = indexgenerate(self.titles, embd, reg,
                                                 batch, topic, rho, test=False)
            self.save()
        else:
            # pdb.set_trace()
            self.parameter(sqlquery(self.paratb))
            print('optimal parameters are', self.para)
            print('loading model...')
            # if True:
            self.load()

    def runyear(self, force_train=False):
        """the main model with cross validation year by year version"""
        self.readdata()
        writeallparas(self.paratb)
        result = sqlquery(self.paratb, cv=None, query=False)
        # pdb.set_trace()
        if len(result) > 0:
            for para in result:
                print('\n{}'.format(datetime.now()))
                # pdb.set_trace()
                self.parameter(para)  # change the state of self.para
                loss = torch.zeros(1, device=device)
                for year in range(self.start, self.end + 1):
                    print(f'training year {year}')
                    loss += indexgenerate(self.readdatayear(year),
                                          self.para['embd'],
                                          self.para['reg'], self.para['batch'],
                                          self.para['topic'], self.para['rho'],
                                          test=True).detach()
                if loss.device == torch.device('cpu'):
                    err = float(loss)
                else:
                    err = float(loss.cpu())
                self.para.update({'loss': err})
                sqlquery(self.paratb, cv=self.para, query=False)
            print('running done for current parameters')
            embd, reg, batch, topic, rho = sqlquery(self.paratb)
            print('embd, reg, batch, topic, rho are chosen optimally as:',
                  embd, reg, batch, topic, rho)
            for year in range(self.start, self.end + 1):
                print(f'training year {year}')
                self.basis, self.lbd = indexgenerate(self.readdatayear(year),
                                                     embd, reg, batch, topic, rho,
                                                     test=False)
                self.save(year)
        elif len(result) > 1 & force_train:
            embd, reg, batch, topic, rho = sqlquery(self.paratb)
            print('force to train again')
            for year in range(self.start, self.end + 1):
                print(f'training year {year}')
                self.basis, self.lbd = indexgenerate(self.readdatayear(year),
                                                     embd, reg, batch, topic, rho,
                                                     test=False)
                self.save(year)

        else:
            # pdb.set_trace()
            self.parameter(sqlquery(self.paratb))
            print('optimal parameters are', self.para)
            print('loading model...')
            self.load()

    def save(self, year=None):
        """save the model"""
        if year == None:
            with open(savefolder + 'model.wig', 'wb') as f:
                pickle.dump([self.basis, self.lbd, self.data, self.tb, self.start,
                             self.end, self.dtype, self.period, self.titles], f)
            print('saving successful')
        else:
            datapath = savefolder + 'year/'
            if not os.path.exists(datapath):
                os.mkdir(datapath)
            with open(datapath + f'model_{year}.wig', 'wb') as f:
                pickle.dump([self.basis, self.lbd, self.data, self.tb, self.start,
                             self.end, self.dtype, self.period, self.titles], f)
            print('saving successful')

    def load(self, year=None):
        """load the model"""
        if year == None:
            print('loading basis and weight')
            with open(savefolder + 'model.wig', 'rb') as f:
                self.basis, self.lbd, self.data, self.tb, \
                    self.start, self.end, self.dtype, \
                    self.period, self.titles = pickle.load(f)
        else:
            datapath = savefolder + 'year/'
            print('loading basis and weight')
            with open(datapath + f'model_{year}.wig', 'rb') as f:
                self.basis, self.lbd, self.data, self.tb, \
                    self.start, self.end, self.dtype, \
                    self.period, self.titles = pickle.load(f)

    def generate(self, save=True, datapath='./data/'):
        if self.mode == 'all':
            return self.generateall(save, datapath)
        elif self.mode == 'year':
            return self.generateyear(save, datapath)

    def generateall(self, save, datapath):
        """generate index by svd for current basis and lbd"""
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
        basis = self.basis.cpu().numpy()
        lbd = self.lbd.cpu().numpy()
        # pdb.set_trace()
        b_mu = basis.mean(0)
        b_std = basis.std(0)
        b_scale = (basis - b_mu) / b_std  # SVD does not requires scaling
        basis_svd = svd.fit_transform(basis.T).T
        l1, l2 = zip(*self.titles.items())
        # pdb.set_trace()
        ym = np.array(l1).reshape(-1, 1)
        monlen = np.array(list(map(len, l2)))
        index = basis_svd @ lbd
        index_sliced = slicearray(index, monlen)
        index_sum = np.array(
            list(map(lambda x: x.sum(), index_sliced))).reshape(-1, 1)
        index_mean = np.array(
            list(map(lambda x: x.mean(), index_sliced))).reshape(-1, 1)
        m = np.concatenate([ym, index_sum, index_mean], axis=1)
        df = pd.DataFrame(m, columns=['yearmon', 'svdsum', 'svdmean'])
        if save:
            if not os.path.exists(datapath):
                os.mkdir(datapath)
            with open(datapath + 'data.tsv', 'w') as f:
                df.to_csv(f, sep='\t', header=True, index=False)
        return df

    def generateyear(self, save, datapath):
        """generate index by svd for current basis and lbd"""
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=2)
        df = pd.DataFrame(columns=['yearmon', 'svdsum', 'svdmean'])
        for year in range(self.start, self.end + 1):
            self.load(year)
            basis = self.basis.cpu().numpy()
            lbd = self.lbd.cpu().numpy()
            # pdb.set_trace()
            # b_mu = basis.mean(0)
            # b_std = basis.std(0)
            # b_scale = (basis - b_mu) / b_std  # SVD does not requires scaling
            basis_svd = svd.fit_transform(basis.T).T
            tmpdict = {i: self.titles[i] for i in timeparse(year, self.period)}
            l1, l2 = zip(*tmpdict.items())  # WARNING: titles are for all
            # pdb.set_trace()
            ym = np.array(l1).reshape(-1, 1)
            monlen = np.array(list(map(len, l2)))
            index = basis_svd @ lbd
            index_sliced = slicearray(index, monlen)
            index_sum = np.array(
                list(map(lambda x: x.sum(), index_sliced))).reshape(-1, 1)
            index_mean = np.array(
                list(map(lambda x: x.mean(), index_sliced))).reshape(-1, 1)
            m = np.concatenate([ym, index_sum, index_mean], axis=1)
            df2 = pd.DataFrame(m, columns=['yearmon', 'svdsum', 'svdmean'])
            df = df.append(df2, ignore_index=True, sort=False)
        if save:
            if not os.path.exists(datapath):
                os.mkdir(datapath)
            with open(datapath + 'data_year.tsv', 'w') as f:
                df.to_csv(f, sep='\t', header=True, index=False)
        return df

    def plot(self, plotpath='./plot/'):
        """plot after generating index"""
        if self.mode == 'all':
            pass
        elif self.mode == 'year':
            plotpath = plotpath + 'year/'
        if not os.path.exists(plotpath):
            os.mkdir(plotpath)
        dfsvd = self.generate()
        dfsvd['yearmon'] = pd.to_datetime(dfsvd['yearmon'], format='%Y-%m')
        dfsvd = dfsvd.set_index('yearmon')
        dfo = pd.read_csv('uncer.tsv', sep='\t', header=0)
        dfo['month'] = pd.to_datetime(dfo['month'], format='%Y-%m')
        dfo = dfo.set_index('month')
        dfa = pd.read_csv('Data_in_Brief_LDA_EPU.csv', sep=',', header=0)
        dfa['Date'] = pd.to_datetime(dfa['Date'], format='%b-%y')
        dfa = dfa.set_index('Date')

        df = pd.concat([dfsvd, dfo['news_index'], dfa['EPU1']],
                       axis=1, sort=True).reindex(dfsvd.index)
        # pdb.set_trace()
        # df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
        df_noarz = df.loc['1985':'2018', :].copy()
        df = df.loc['1989':'2016-08', :].copy()
        df_noarz['svdsum'] = scale(df_noarz['svdsum']) + 100
        df_noarz['news_index'] = scale(df_noarz['news_index']) + 100
        df_noarz.rename(columns={'news_index': 'news'}, inplace=True)
        # dfall['ica'][0]
        # pdb.set_trace()
        df['svdsum'] = scale(df['svdsum']) + 100
        df['svdmean'] = scale(df['svdmean']) + 100
        df['news_index'] = scale(df['news_index']) + 100
        df['EPU1'] = scale(df['EPU1']) + 100
        df.rename(columns={'news_index': 'news', 'EPU1': 'arz'}, inplace=True)
        with open(plotpath + 'scaledindex.tsv', 'w') as f:
            df.to_csv(f, sep='\t', header=True, index=True)

        p = df_noarz.plot(y=['news', 'svdsum'], linewidth=1.0,
                          figsize=(17, 10), x_compat=True, rot=0)
        p.xaxis.set_major_locator(mdates.YearLocator())  #
        p.xaxis.set_major_formatter(years_fmt)
        # p.xaxis.set_label_coords(0.01, 0)
        p.set_xlabel('Time (Monthly)')
        p.set_ylabel('Index')
        p.set_title('Economic Policy Uncertainty Index (EPU)')
        p.legend(['EPU', 'EPU_WIG'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'orisvdepu.png', dpi=300, bbox_inches='tight')

        # pdb.set_trace()
        # draw the plots
        p = df.plot(y=['news', 'svdsum', 'arz'], linewidth=1.0,
                    figsize=(17, 10), x_compat=True, rot=0)
        p.xaxis.set_major_locator(mdates.YearLocator())  #
        p.xaxis.set_major_formatter(years_fmt)
        p.set_xlabel('Time (Monthly)')
        p.set_ylabel('Index')
        p.set_title('Economic Policy Uncertainty Index (EPU)')
        p.legend(['EPU', 'EPU_WIG', 'EPU_LDA'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'newsvdplot.png', dpi=300, bbox_inches='tight')

        # TODO: draw other plots
        dfprev = pd.read_csv('../data7svd/icapcascale.tsv', sep='\t', header=0)
        dfprev = dfprev.set_index('yearmon')
        df['svdprev'] = dfprev['svd']
        p = df.plot(y=['news', 'svdsum', 'svdprev', 'arz'], linewidth=1.0,
                    figsize=(17, 10), x_compat=True, rot=0)
        # locator = mdates.YearLocator()
        # formatter = mdates.ConciseDateFormatter(locator)
        p.xaxis.set_major_locator(mdates.YearLocator())
        p.xaxis.set_major_formatter(years_fmt)
        p.set_xlabel('Time (Monthly)')
        p.set_ylabel('Index')
        p.set_title('Monthly Comparison with Previous Results')
        p.legend(['EPU', 'EPU_WIG', 'EPU_WIGprev', 'EPU_LDA'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'prevsvdplot.png', dpi=300, bbox_inches='tight')

        dfall = pd.read_csv('./plot/year/scaledindex.tsv', sep='\t', header=0)
        dfall = dfall.set_index('yearmon')
        df['svdyear'] = dfall['svdsum']
        p = df.plot(y=['news', 'svdsum', 'svdyear', 'arz'], linewidth=1.0,
                    figsize=(17, 10), x_compat=True, rot=0)
        p.xaxis.set_major_locator(mdates.YearLocator())
        p.xaxis.set_major_formatter(years_fmt)
        p.set_xlabel('Month')
        p.set_ylabel('Index')
        # p.set_title('Monthly Comparison')
        p.legend(['EPU', 'EPU_WIGall', 'EPU_WIGyear',
                  'EPU_LDA'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'yearallsvdplot.png',
                    dpi=300, bbox_inches='tight')

        cycle, trend = sm.tsa.filters.hpfilter(df.news, 129600)
        epu_decomp = df[['news']]
        epu_decomp['news_trend'] = trend.values
        epu_decomp['news_cycle'] = cycle.values

        svdcycle, svdtrend = sm.tsa.filters.hpfilter(df.svdsum, 129600)
        epu_decomp['svd'] = df[['svdsum']]
        epu_decomp['svd_trend'] = svdtrend.values
        epu_decomp['svd_cycle'] = svdcycle.values
        # p = epu_decomp

        arzcycle, arztrend = sm.tsa.filters.hpfilter(df.arz, 129600)
        epu_decomp['arz'] = df[['arz']]
        epu_decomp['arz_trend'] = arztrend.values
        epu_decomp['arz_cycle'] = arzcycle.values

        p = epu_decomp.plot(y=['news', 'news_trend', 'svd_trend', 'arz_trend'], linewidth=1.0,
                            figsize=(17, 10), x_compat=True, rot=0)
        p.xaxis.set_major_locator(mdates.YearLocator())
        p.xaxis.set_major_formatter(years_fmt)
        p.set_xlabel('Month')
        # p.set_ylabel('Index')
        # p.set_title('Monthly Comparison')
        p.legend(['EPU', 'EPU_trend', 'WIG_trend',
                  'LDA_trend'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'EPU_trend.png', dpi=200, bbox_inches='tight')

        df['svddiff'] = abs(df['svdsum'] - df['news'])
        # df['icadiff'] = abs(df['ica'] - df['news'])
        # df['pcadiff'] = abs(df['pca'] - df['news'])
        df['arzdiff'] = abs(df['arz'] - df['news'])
        df['svddiffcs'] = np.cumsum(df['svddiff'].values)
        # df['icadiffcs'] = np.cumsum(df['icadiff'].values)
        # df['pcadiffcs'] = np.cumsum(df['pcadiff'].values)
        df['arzdiffcs'] = np.cumsum(df['arzdiff'].values)

        p = df.plot(y=['svddiffcs', 'arzdiffcs'], linewidth=1.0,
                    figsize=(17, 10), x_compat=True, rot=0)
        p.xaxis.set_major_locator(mdates.YearLocator())
        p.xaxis.set_major_formatter(years_fmt)
        p.set_xlabel('Time (Monthly)')
        p.set_ylabel('Index')
        p.set_title('Cumulated Differences between WIG and LDA')
        p.legend(['WIG', 'LDA'], loc='upper left')
        fig = p.get_figure()
        fig.savefig(plotpath + 'cumsumdiff.png', dpi=200, bbox_inches='tight')

        print('pearson svd', pearsonr(epu_decomp['svd'], epu_decomp['news']))
        print('pearson arz', pearsonr(epu_decomp['arz'], epu_decomp['news']))

        print('pearson svd trend', pearsonr(
            epu_decomp['svd_trend'], epu_decomp['news_trend']))
        print('pearson arz trend', pearsonr(
            epu_decomp['arz_trend'], epu_decomp['news_trend']))

        print('pearson svd cycle', pearsonr(
            epu_decomp['svd_cycle'], epu_decomp['news_cycle']))
        print('pearson arz cycle', pearsonr(
            epu_decomp['arz_cycle'], epu_decomp['news_cycle']))

        print('spearsonr svd', spearmanr(df['svdsum'], df['news']))
        print('spearsonr arz', spearmanr(df['arz'], df['news']))

        print('spearmanr svd trend', spearmanr(
            epu_decomp['svd_trend'], epu_decomp['news_trend']))
        print('spearmanr arz trend', spearmanr(
            epu_decomp['arz_trend'], epu_decomp['news_trend']))

        print('spearmanr svd cycle', spearmanr(
            epu_decomp['svd_cycle'], epu_decomp['news_cycle']))
        print('spearmanr arz cycle', spearmanr(
            epu_decomp['arz_cycle'], epu_decomp['news_cycle']))
        pass


def pplot():
    df1 = pd.read_csv('./data/data.tsv', sep='\t', header=0)
    df2 = pd.read_csv('./data/data_year.tsv', sep='\t', header=0)


def slicearray(unsliced, indexlist):
    """slice a whole array into list of subarrays

    works for x * D dimension array, x1 * d1 + ... + xn * dn
    """
    # assert len(indexlist) != 0
    assert len(indexlist) != 0
    slicer = np.cumsum(np.asarray(indexlist))
    if len(unsliced.shape) >= 2:
        sliced = [unsliced[:, slicer[i]:slicer[i + 1]]
                  for i in range(len(slicer) - 1)]
        sliced.insert(0, unsliced[:, :slicer[0]])
        # check if sliced match the original
        assert np.any(np.concatenate(sliced, axis=1) == unsliced) is not False
    elif len(unsliced.shape) == 1:
        sliced = [unsliced[slicer[i]:slicer[i + 1]]
                  for i in range(len(slicer) - 1)]
        sliced.insert(0, unsliced[:slicer[0]])

    return sliced


def slicetensor(unsliced, indexlist):
    """slice a whole array into list of subarrays

    works for x * D dimension array, x1 * d1 + ... + xn * dn
    now support d >= 1
    """
    # pdb.set_trace()
    # indexlist = torch.tensor(indexlist)
    assert len(indexlist) != 0
    slicer = torch.as_tensor(indexlist).cumsum(0)
    if len(unsliced.shape) >= 2:
        sliced = [unsliced[:, slicer[i]:slicer[i + 1]]
                  for i in range(len(slicer) - 1)]
        # check if sliced match the original
        sliced.insert(0, unsliced[:, :slicer[0]])
        assert bool(torch.any(torch.cat(sliced, axis=1)
                              == unsliced)) is not False
    elif len(unsliced.shape) == 1:
        sliced = [unsliced[slicer[i]:slicer[i + 1]]
                  for i in range(len(slicer) - 1)]
        sliced.insert(0, unsliced[:slicer[0]])
        assert bool(torch.any(torch.cat(sliced) == unsliced)) is not False
    return sliced


def slicelist(unsliced, indexlist):
    slicer = torch.as_tensor(indexlist).cumsum(0)
    sliced = [unsliced[slicer[i]:slicer[i + 1]]
              for i in range(len(slicer) - 1)]
    sliced.insert(0, unsliced[:slicer[0]])
    # assert sum(map(len, sliced)) == len(unsliced)
    return sliced


def titlelist2id(titlelist, startnum):
    idlist = list(range(startnum, startnum + len(titlelist) + 1))
    assert len(idlist) == len(titlelist)
    return idlist


def writeallparas(tablename):
    conn = lite.connect('parameters.sqlite')
    c = conn.cursor()
    c.execute("""
                CREATE TABLE IF NOT EXISTS `{}` (
                    `embd`  INT,
                    `reg`  REAL,
                    `batch` INT,
                    `topic` INT,
                    `rho` REAL,
                    `loss` REAL,
                    PRIMARY KEY(`embd`, `reg`,`batch`, `topic`, `rho`));
            """.format(tablename))
    for paras in cvparas():
        assert len(paras) == 5
        # pdb.set_trace()
        c.execute("""
                    INSERT OR IGNORE INTO `{}` (embd, reg, batch, topic, rho)
                    VALUES (?,?,?,?,?);""".format(tablename), paras)
    conn.commit()
    conn.close()
    pass


def cvparas():
    embdlist = [10, 20]
    reglist = [.1]
    batchlist = [32, 64]
    topiclist = [4, 8]
    rholist = [0.005]
    for reg in reglist:
        for batch in batchlist:
            for topic in topiclist:
                for embd in embdlist:
                    for rho in rholist:
                        yield tuple([embd, reg, batch, topic, rho])


def sqlquery(paratable, cv=None, query=True):
    conn = lite.connect('parameters.sqlite')
    c = conn.cursor()
    if cv is None and query:  # find min value, default mode
        c.execute("""
        SELECT * FROM `{}` ORDER BY loss ASC LIMIT 1
        """.format(paratable))
        embd, reg, batch, topic, rho, loss = c.fetchone()
        # conn.commit()
        conn.close()
        return embd, reg, batch, topic, rho
    elif cv is not None and query:  # check current parameters in sql
        embd, reg, batch = cv['embd'], cv['reg'], cv['batch']
        topic, rho, loss = cv['topic'], cv['rho'], cv['loss']
        paras = tuple([embd, reg, batch, topic, rho])
        # pdb.set_trace()
        c.execute("""
        SELECT * FROM `{}` WHERE (embd, reg, batch, topic, rho) =(?,?,?,?,?)
        """.format(paratable), paras)
        result = c.fetchall()
        conn.close()
        return True if len(result) >= 1 else False
    elif cv is not None and not query:  # update bound NULL -> value
        embd, reg, batch = cv['embd'], cv['reg'], cv['batch']
        topic, rho, loss = cv['topic'], cv['rho'], cv['loss']
        paras = tuple([loss, embd, reg, batch, topic, rho])
        c.execute("""
        UPDATE `{}` SET loss = (?)
        WHERE (embd, reg, batch, topic, rho)=(?,?,?,?,?);""".format(paratable), paras)
        conn.commit()
        conn.close()
        pass
    elif cv is None and not query:  # if exist NULL bound
        c.execute("""
        SELECT * FROM `{}` WHERE loss IS NULL;
        """.format(paratable))
        result = c.fetchall()
        conn.close()
        return result


def tokenize(title_dict):
    '''tokenizer text, return dict of vocab and loss matrix'''
    # lemm = WordNetLemmatizer()
    nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser"])
    # add merge ents to pipeline
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    docs = [v for k, v in title_dict.items()]
    texts = [item for doc in docs for item in doc]
    lemmdocs = [tuple([re.sub(r'[^\w\s]', '', t.lemma_)
                       for t in doc if re.sub(r'[^\w\s]', '', t.lemma_).replace(' ', '')])
                for doc in nlp.pipe(texts, batch_size=100)]
    del nlp
    del merge_ents
    del docs
    del texts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lemmdocs


def docembd(docs, embd, batch_words=500, metric='sqeuclidean', dtype=torch.float32):

    model = Word2Vec(docs, min_count=1, sg=1, size=embd,
                     batch_words=batch_words, workers=10, iter=5)
    w_vecs = model.wv
    # get euclidean metric from embedding
    X = torch.tensor(w_vecs[w_vecs.vocab], dtype=dtype, device=device)
    # pdb.set_trace()
    M = dist(X, metric=metric)
    # M /= M.max()  # rescale to unit
    del model
    # M = torch.tensor(M, dtype=dtype, device=device)
    return docs, list(w_vecs.vocab), mjit(M)


def doctokenize(title_dict, embd):
    docs = tokenize(title_dict)
    return docembd(docs, embd)


def getfreq(vocab_list, sentence):
    """get the word distribution of a certain sentence"""
    assert sentence != []
    senvec = list(map(lambda x: sentence.count(x), vocab_list))
    sendist = torch.tensor(senvec, dtype=torch.float32)
    # pdb.set_trace()
    return tensordiv(sendist)


def euclidean_distances(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    This version is the pytorch implementation of POT ot.dist function
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """
    # # WARNING: check tensor memory: a.element_size() * a.nelement()
    # pdb.set_trace()
    XX = torch.einsum('ij,ij->i', X, X).unsqueeze(1)
    YY = torch.einsum('ij,ij->i', Y, Y).unsqueeze(0)
    distances = torch.matmul(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    torch.max(distances, torch.zeros(
        distances.shape, device=device), out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances = distances - torch.diag(distances.diag())
    return distances if squared else torch.sqrt(distances, out=distances)


def euclidean_distances_another(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    Another version (takes even more memory than POT one).
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """
    # # WARNING: check tensor memory: a.element_size() * a.nelement()
    n = X.size(0)
    m = Y.size(0)
    d = X.size(1)

    X = X.unsqueeze(1).expand(n, m, d)
    Y = Y.unsqueeze(0).expand(n, m, d)
    if squared:
        dist = torch.pow(X - Y, 2).sum(2)
    else:
        dist = torch.pow(X - Y, 1).sum(2)
    # if X is Y:
    #     # Ensure that distances between vectors and themselves are set to 0.0.
    #     # This may not be the case due to floating point rounding errors.
    #     distances = distances - torch.diag(distances.diag())
    return dist


def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
    Parameters
    ----------
    x1 : ndarray, shape (n1,d)
        matrix with n1 samples of size d
    x2 : array, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | callable, optional
        default to be squared euclidean distance, but normal euclidean dist
        is also available. Other type of distances are not supported yet.
    Returns
    -------
    M : np.array (n1,n2)
        distance matrix computed with given metric
    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    # return cdist(x1, x2, metric=metric)


def strreplace(string):
    """replace html code and others in string to clean dataset"""
    s = html.unescape(string)
    s = unidecode(s)
    # s = s.replace("'s", ' ')
    # s = s.replace("'", ' ')
    # s = s.replace('""', ' ')
    # s = re.sub(r'[^\w\s]', ' ', s)
    # s = re.sub(r'[0-9]+', ' ', s)
    return s


def timeparse(year, period):
    """
    determine if the model should built on yearly, monthly or daily step
    e.g. if 'Y', then '2018'; if 'M', then '2018-01'; if 'D', then '2018-01-01'

    period = 'Y', 'M', 'D'

    return list of %Y-%m-%d
    """
    if period == 'Y':
        return [f'{year}']
    elif period == 'M':
        ym_par = partial(lambda xy, xmon: str(datetime(xy, xmon, 1))[0:7], year)
        return list(map(ym_par, range(1, 13)))
    elif period == 'D':
        return [str(datetime(year, mon, day)).split(' ')[0]
                for mon in range(1, 13) for day in range(1, mr(year, mon)[1])]


def readsql(datafile, tablename, year):
    """return a generator in case the results are too many"""
    conn = lite.connect(datafile)
    c = conn.cursor()
    query = f"SELECT * FROM {tablename} WHERE date >= (?) AND date <= (?);"
    results = c.execute(query, (str(year), str(year + 1))).fetchall()
    conn.close()
    return results


@torch.jit.script
def mjit(M: Tensor):
    return M / M.max()


@torch.jit.script
def tensordiv(sendist: Tensor):
    return torch.div(sendist, sendist.sum())


@torch.jit.script
def sinkjit(b, lbd, C: Tensor, u: Tensor, v: Tensor, yhat: Tensor):
    """jit computation in main sinkhorn loop"""
    Cu = torch.matmul(C, u)
    v = torch.matmul(C.transpose(0, 1), torch.div(b, Cu))
    yhat = torch.prod(torch.pow(v, lbd.transpose(0, 1)), dim=1).reshape(-1, 1)
    u = torch.div(yhat, v)  # yhat / phi
    return C, Cu, u, v, yhat


@torch.jit.script
def expjit(M: Tensor, reg: Tensor):
    """C: energy function"""
    return torch.exp(- M / reg)


# @timer
def sinkloss_torch(a, M, b, lbd, reg, numItermax=1000, stopThr=1e-9,
                   dtype=torch.float32, verbose=False):
    '''
    parameters:
    a: source dist (1 doc),
    b: target dist (k topics),
    lbd: weights of 1 doc -> k topics,
    M: underlying distance to be distilled
    '''
    # torch.reset_default_graph()
    # init data
    Nini = a.shape[0]
    Nfin = b.shape[0]
    nbb = b.shape[1]
    # init u and v
    u = torch.ones((Nini, nbb), dtype=dtype, device=device)
    v = torch.ones((Nfin, nbb), dtype=dtype, device=device)

    C = expjit(M, reg)

    yhat = torch.zeros(a.shape, dtype=dtype, device=device)
    cpt = 0
    err = 1.

    # forward loop update
    while (err > stopThr and cpt < numItermax):
        uprev = u  # b^{l} = u = beta
        vprev = v
        yhatprev = yhat

        C, Cu, u, v, yhat = sinkjit(b, lbd, C, u, v, yhat)

        if (torch.any(Cu == 0) or torch.any(torch.isnan(u)) or
            torch.any(torch.isinf(u)) or torch.any(torch.isnan(v)) or
            torch.any(torch.isinf(v)) or torch.any(torch.isnan(yhat)) or
                torch.any(torch.isinf(yhat))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            yhat = yhatprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            lhs = torch.sum((u - uprev)**2) / torch.sum((u)**2)
            rhs = torch.sum((v - vprev)**2) / torch.sum((v)**2)
            err = lhs + rhs
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    del C
    del Cu
    del u
    del v
    del yhatprev
    del uprev
    del vprev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loss = torch.norm(yhat - a) ** 2
    # print(loss)
    # pdb.set_trace()
    return loss


# @timer
def sinklossgrad_torch(sa: Tensor, M: Tensor, tb: Tensor, lbd: Tensor, reg: Tensor):
    loss = sinkloss_torch(sa, M, tb, lbd, reg)
    loss.backward()
    with torch.no_grad():
        gradb, gradd = tb.grad, lbd.grad
    return gradb, gradd


def adam(grad, t, m=None, v=None, lr=0.001, eps=1e-8, beta1=0.9, beta2=0.999):
    """implement adam optimizer"""
    # pdb.set_trace()
    if t == 1:
        m = torch.zeros(grad.shape, dtype=torch.float32, device=device)
        v = torch.zeros(grad.shape, dtype=torch.float32, device=device)
    mt = beta1 * m + (1. - beta1) * grad
    vt = beta2 * v + (1. - beta2) * (grad ** 2)
    mt_hat = mt / (1. - beta1 ** t)
    vt_hat = vt / (1. - beta2 ** t)
    grad_adam = lr * mt_hat / (torch.sqrt(vt_hat) + eps)
    return grad_adam, mt, vt


def distillwasser(data, train_id, ten2id, M, reg, batch, topic, rho, repeat=1, optim_method='adam'):
    ''' main model, other training parameters to be added'''
    # here docs is the flat docs returned from previous func
    dl = DataLoader(data, batch_size=batch, shuffle=False, num_workers=10)
    R = torch.randn((M.shape[0], topic), device=device)
    A = torch.randn((topic, len(ten2id)), device=device)
    basis = softmax(R, dim=0)  # softmax over columns
    lbd = softmax(A, dim=0)
    # pdb.set_trace()

    t = 0
    for _ in range(repeat):
        for b_id, batch_data in enumerate(dl):
            for sub_id, each in enumerate(batch_data):
                # pdb.set_trace()
                if torch.cuda.is_available():
                    csa = each.reshape(-1, 1).cuda()
                else:
                    csa = each.reshape(-1, 1)
                # pdb.set_trace()
                clbd = lbd[:, train_id[ten2id[ten2key(
                    each)]]].reshape(-1, 1).contiguous()
                basis.requires_grad_()
                clbd.requires_grad_()
                reg = torch.tensor([reg], device=device)

                # pdb.set_trace()
                gradb, gradd = sinklossgrad_torch(csa, M, basis, clbd, reg)
                t += 1
                if optim_method.lower() == 'adam':
                    if t == 1:
                        gradb_adam, mb, vb = adam(gradb, t, lr=rho)
                        gradd_adam, md, vd = adam(gradd, t, lr=rho)
                    else:
                        gradb_adam, mb, vb = adam(gradb, t, mb, vb, lr=rho)
                        gradd_adam, md, vd = adam(gradd, t, md, vd, lr=rho)

                    R = R - gradb_adam
                    A = A - gradd_adam
                elif optim_method.lower() == 'sgd':
                    R = R - rho * gradb
                    A = A - rho * gradd
                # if t % 100:
                #     print(f'{t:5d}')
            basis = softmax(R, dim=0)
            lbd = softmax(A, dim=0)
            # print('End of a batch\n')
        # pdb.set_trace()
    del R
    del A
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return basis, lbd


def distillwassertest(data, test_id, M, reg, basis, lbd):
    '''given a year's doc, return loss under current basis and weights'''
    # docs, vocab_list, M = doctokenize(title_dict, embd)
    dl = DataLoader(data, batch_size=1, shuffle=False, num_workers=10)
    err = torch.zeros(1, device=device)
    reg = torch.tensor([reg], device=device)
    # pdb.set_trace()
    # random.shuffle(docs_test)
    for b_id, batch in enumerate(dl):
        if torch.cuda.is_available():
            sa = batch.reshape(-1, 1).cuda()
        else:
            sa = batch.reshape(-1, 1)
        clbd = lbd[:, test_id[b_id]].reshape(-1, 1).contiguous()
        loss = sinkloss_torch(sa, M, basis, clbd, reg).detach()
        err += loss
        # err.append(loss)
    return err


def datasplit(input, ratio):
    """split dataset into two parts, with shuffle by pytorch
    Parameters
    ----------
    input: data need to be splitted
    ratio: ratio of size of validation set
    """
    totallen = len(input)
    x = math.floor(totallen * ratio)
    return random_split(input, [totallen - x, x])


def doc2dist(docs, vocab, core_num):
    """cast docs into distributions by multiprocessing
    Parameters
    ----------
    docs: input data in list of tokens
    vocab: list of all tokens in all docs
    core_num: number of threads
    """
    with torch.multiprocessing.Pool(processes=core_num) as p:
        results = p.map(partial(getfreq, vocab), docs)
    return results


def ten2key(tensor: Tensor):
    """turn a dense tensor into its sparse format and form a tuple as dict keys

    e.g. [1., 0., 0., 0., 2.] => ((1, 1.), (5, 2.))

    current implementation only works for 1-d tensor, when n >= 2,
    use tuple(list(map(lambda x: tuple(x.numpy().tolist()), tensor.nonzero())))
    """
    return tuple([(int(i), float(tensor[i])) for i in tensor.nonzero()])


# def preprocess(titles, embd, worker):
#     docs, vocab, M = doctokenize(titles, embd)
#     predocs = doc2dist(docs, vocab, worker)
#     # id2ten = {i: predocs[i] for i in range(len(docs))}
#     # doc2id = {docs[i]: i for i in range(len(docs))}
#     ten2id = {ten2key(predocs[i]): i for i in range(len(docs))}
#     return docs, vocab, M, predocs, ten2id  # , id2ten


def old2new(newdata, ten2id):
    """create new dict, key: old id, value: new id"""
    return {ten2id[ten2key(newdata[i])]: i for i in range(len(newdata))}


def new2old(newdata, ten2id):
    """key: new id, value: old id"""
    return {i: ten2id[ten2key(newdata[i])] for i in range(len(newdata))}


@timer
def indexgenerate(titles, embd=48, reg=.01, batch=16, topic=8, rho=.01, repeat=1,
                  test=False, worker=mp.cpu_count() - 1):
    """main function for gen or test"""
    docs, vocab, M = doctokenize(titles, embd)
    predocs = doc2dist(docs, vocab, worker)
    ten2id = {ten2key(predocs[i]): i for i in range(len(docs))}
    # docs bijection with predocs, since mp.map does not change order
    data = WigDataset(predocs)
    assert len(docs) == len(ten2id)
    # pdb.set_trace()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # pin_memory=True, collate_fn=collate_par)
    print(f'{len(docs)} docs, {len(vocab)} vocab')
    # pdb.set_trace()
    if test:
        docs_train, docs_test = datasplit(data, ratio=0.33)
        # pdb.set_trace()
        train_id = old2new(docs_train, ten2id)
        test_id = new2old(docs_test, ten2id)
        basis, lbd = distillwasser(data=docs_train, train_id=train_id, ten2id=ten2id,
                                   M=M, reg=reg, batch=batch,
                                   topic=topic, rho=rho, repeat=repeat)
        loss = distillwassertest(docs_test, test_id, M, reg, basis, lbd)
        # pdb.set_trace()
        return loss
    else:
        # docs = datasplit(data, ratio=1)[0]
        train_id = old2new(data, ten2id)
        basis, lbd = distillwasser(data=data, train_id=train_id, ten2id=ten2id,
                                   M=M, reg=reg, batch=batch,
                                   topic=topic, rho=rho, repeat=repeat)
        return basis.detach(), lbd.detach()


def main():
    # pplot()
    # pdb.set_trace()
    # first run, all years together
    model = WIG(datafile='headlines.sqlite',
                tablename='headlines_reduce',
                paratb='parameters_allreduce',
                mode='all',
                summary=False)
    model.run(force_train=False)
    model.plot()
    pass


if __name__ == "__main__":
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    main()
