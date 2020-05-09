import multiprocessing as mp
import os
import pdb
from collections import OrderedDict, defaultdict
from datetime import datetime
from functools import partial
from operator import itemgetter

import numpy as np
import pandas as pd
import spacy
import torch
from gensim.models import Word2Vec
# from scipy.stats import pearsonr, spearmanr
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import scale
from torch import Tensor, optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from model import WasserIndexGen
from utils import timer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy.prefer_gpu()

# from nltk.corpus import stopwords;  stopwords.words('english');
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
              'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
              "she's",  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
              'they',  'them', 'their', 'theirs', 'themselves', 'what', 'which',
              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
              'through', 'during', 'before', 'after', 'above', 'below', 'to',
              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
              'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
              'can', 'will', 'just', 'don', "don't", 'should', "should've",
              'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
              "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
              "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
              "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
              "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
              "won't", 'wouldn', "wouldn't"]


class WIG():
    def __init__(self,
                 dataset,
                 train_test_ratio=[0.7, 0.3],
                 emsize=10,
                 batch_size=64,
                 num_topics=4,
                 reg=0.1,
                 epochs=5,
                 opt='adam',
                 lr=0.005,
                 wdecay=1.2e-6,
                 log_interval=50,
                 seed=0,
                 prune_topk=0,
                 l1_reg=0.01,
                 n_clusters=10,
                 ckpt_path='./ckpt',
                 numItermax=1000,
                 stopThr=1e-9,
                 dtype=torch.float32,
                 spacy_model='en_core_web_sm',
                 metric='sqeuclidean',
                 merge_entity=True,
                 process_fn=None,
                 remove_stop=False,
                 remove_punct=True,
                 device='cuda',
                 interval='M',
                 visualize_every=1,
                 loss_per_batch=False,
                 **kwargs):
        """
        Parameters:
        ======
        dataset         : list, of (date, doc) pairs
        train_test_ratio : list, of floats sum to 1, how to split dataset
        emsize          : int, dim of embedding
        batch_size      : int, size of a batch
        num_topics      : int, K topics
        reg             : float, entropic regularization term in Sinkhorn
        epochs          : int, epochs to train
        lr              : float, learning rate for optimizer
        wdecay          : float, L-2 regularization term used by some optimizers
        log_interval    : int, print log one per k steps
        seed            : int, pseudo-random seed for pytorch
        prune_topk      : int, max no of tokens to use for pruning vocabulary
        l1_reg          : float, L1 penalty for pruning
        n_clusters      : int, KMeans clusters
        opt             : str, which optimizer to use, default to 'adam'
        ckpt_path       : str, checkpoint when training model
        numItermax      : int, max steps to run Sinkhorn, dafault 1000
        dtype           : torch.dtype, default torch.float32
        spacy_model     : str, spacy language model name
                        Default: nlp = spacy.load(
                            'en_core_web_sm', disable=["tagger"])
        metric          : str, 'sqeuclidean' or 'euclidean'
        merge_entity    : bool, merge entity detected by spacy model, default True
        remove_stop     : bool, whether to remove stop words, default False
        remove_punct    : bool, whether to remove punctuation, default True
        interval        : 'M', 'Y', 'D'
        visualize_every : int,
        loss_per_batch  : bool, if print loss per batch

        Also parameters from Word2Vec
        """

        global dt
        if dtype:
            dt = dtype

        global dev
        if device == 'cuda':
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            dev = torch.device('cpu')

        # hyperparameters
        self.batch_size = batch_size
        self.num_topics = num_topics
        self.reg = reg
        self.epochs = epochs
        self.lr = lr
        self.emsize = emsize

        # opt
        self.opt = opt
        self.lr = lr
        self.wdecay = wdecay
        self.interval = interval
        self.visualize_every = visualize_every
        self.loss_per_batch = loss_per_batch

        self.log_interval = log_interval

        # ckpt
        self.ckpt = os.path.join(ckpt_path,
                                 f'WIG_Bsz_{batch_size}_K_{num_topics}_LR_{lr}_EMsize_{emsize}_Reg_{reg}_Opt_{opt}_CompressTopk_{prune_topk}')
        self.ckpt_path = ckpt_path
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        # set random seed
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        sentences, self.id2date, self.id2doc, self.senid2docid, self.date2idlist = \
            self.idmap(dataset, spacy_model, merge_entity, process_fn,
                       remove_stop, remove_punct)
        # pdb.set_trace()
        # prepare train, eval, and test data
        self.d_l = d_l = len(self.senid2docid.keys())
        assert sum(train_test_ratio) == 1., \
            'Three shares do not sum to one.'
        train_r, test_r = train_test_ratio
        print(f'Splitting data by ratio: {train_r} {test_r}')
        tr_l = round(d_l * train_r)
        ts_l = d_l - tr_l

        # shuffle ids
        data_ids = torch.randperm(d_l)  # ids of splitted sentences

        # Word2Vec model use more workers and less min_count than default
        try:
            workers
        except NameError:
            workers = 20
        try:
            min_count
        except NameError:
            min_count = 1
        # add to kwargs
        kwargs['size'] = emsize
        kwargs['seed'] = seed
        kwargs['workers'] = workers
        kwargs['min_count'] = min_count
        print('run Word2Vec model for word embeddings...')
        word2vec = Word2Vec(sentences=[sentences[i] for i in data_ids],
                            **kwargs)
        wv = word2vec.wv
        self.vocab = wv.vocab
        print(f'Vocab length is {len(wv.vocab)}')

        # choose with algorithm to use, if compressed then shrink vocab dim
        if prune_topk == 0:
            print('using original algorithm')
            X = torch.tensor(wv[wv.vocab], dtype=dtype, device=device)
            self.M = mjit(dist(X, metric=metric))
            mycollate = partial(getfreq_single_original, wv)
        elif prune_topk > 0:
            print('using compressed dictionary algorithm')
            print('compressed dictionary length is {}'.format(prune_topk))
            self.M, id2comdict = compress_dictionary(wv, topk=prune_topk,
                                                     n_clusters=n_clusters,
                                                     l1_reg=l1_reg,
                                                     metric='sqeuclidean')
            mycollate = partial(getfreq_single_compress, wv, id2comdict)
        else:
            # TODO: geodesic regression with l1?
            raise ValueError("use 'original' or 'compress' algotithm")

        tr_ids, ts_ids = data_ids.split([tr_l, ts_l])
        self.tr_ids, self.ts_ids = tr_ids, ts_ids

        # pdb.set_trace()
        self.tr_dl = DataLoader([sentences[i] for i in tr_ids],
                                batch_size=batch_size, collate_fn=mycollate)
        self.ts_dl = DataLoader([sentences[i] for i in ts_ids],
                                batch_size=batch_size, collate_fn=mycollate)

        self.model = WasserIndexGen(batch_size=batch_size,
                                    num_topics=num_topics,
                                    reg=reg,
                                    numItermax=numItermax,
                                    stopThr=stopThr,
                                    dtype=dt,
                                    device=dev)

        print(f'WIG model {self.model}')

        # init
        self.R = torch.randn((self.M.shape[0], self.num_topics),
                             device=dev)
        self.A = torch.randn((self.num_topics, self.d_l),
                             device=dev)
        # softmax over columns
        self.basis = softmax(self.R, dim=0)
        self.lbd = softmax(self.A, dim=0)

    @timer
    def train(self, loss_per_batch=False):
        # set optimizer
        if self.opt == 'adam':
            optimizer = optim.Adam([self.R, self.A], lr=self.lr,
                                   weight_decay=self.wdecay)
        elif self.opt == 'adagrad':
            optimizer = optim.Adagrad([self.R, self.A], lr=self.lr,
                                      weight_decay=self.wdecay)
        elif self.opt == 'adadelta':
            optimizer = optim.Adadelta([self.R, self.A], lr=self.lr,
                                       weight_decay=self.wdecay)
        elif self.opt == 'rmsprop':
            optimizer = optim.RMSprop([self.R, self.A], lr=self.lr,
                                      weight_decay=self.wdecay)
        elif self.opt == 'asgd':
            optimizer = optim.ASGD([self.R, self.A], lr=self.lr,
                                   weight_decay=self.wdecay, t0=0, lambd=0.)
        else:
            print('Optimizer not supported . Defaulting to vanilla SGD...')
            optimizer = optim.SGD([self.R, self.A], lr=self.lr)

        # cnt = 0
        best_loss = 1e9
        for epoch in range(self.epochs):
            total_loss = 0.
            tr_id = 0
            for b_id, batch in enumerate(self.tr_dl):
                for each in batch:
                    self.R.requires_grad_()
                    self.A.requires_grad_()
                    self.basis.requires_grad_()
                    self.lbd.requires_grad_()

                    optimizer.zero_grad()

                    csa = each.view(-1, 1).to(dev)
                    clbd = self.lbd[:,
                                    self.tr_ids[tr_id]].view(-1, 1).to(dev)

                    reg = torch.tensor([self.reg]).to(dev)

                    loss = self.model(csa, self.M, self.basis, clbd, reg)
                    loss.backward()
                    optimizer.step()

                    self.R.detach_()
                    self.A.detach_()
                    self.basis.detach_()
                    self.lbd.detach_()

                    tr_id += 1

                    if not torch.isnan(loss).item():
                        # pdb.set_trace()
                        total_loss += loss.item()

                if loss_per_batch:
                    print('Average Loss: {0:.4f}'.format(total_loss / tr_id))

                # softmax over columns
                self.basis = softmax(self.R, dim=0)
                self.lbd = softmax(self.A, dim=0)

            # if not self.infer:
            # evaluate after training
            eval_loss = self.evaluate(self.model, self.ts_dl, self.ts_ids,
                                      self.basis, self.lbd)
            if eval_loss < best_loss:  # save bast model among all epochs
                with open(self.ckpt, 'wb') as f:
                    torch.save((self.model, self.basis, self.lbd), f)
                best_loss = eval_loss
            print('*' * 50)
            print('Epoch: {}, LR: {}, Train Loss: {:.2f}, Eval Loss: {:.2f}'.format(
                epoch, self.lr, total_loss / tr_id, eval_loss))
            print('*' * 50)
            if epoch % self.visualize_every == 0:
                # TODO: implement visualize function per log_interval
                #     visualize(self.model, self.vocab)
                pass
        with open(self.ckpt, 'rb') as f:
            m, basis, lbd = torch.load(f)
        eval_loss = self.evaluate(m, self.ts_dl, self.ts_ids,
                                  self.basis, self.lbd)
        print(f'Evaluation Loss: {eval_loss.item()}')
        return eval_loss.item()

    def evaluate(self, model, data_loader, data_ids, basis, lbd):
        """

        data_loader: either eval_loader or test_loader
        data_ids: self.eval_ids or self.test_ids
        """
        model.eval()
        with torch.no_grad():
            total_loss = 0.
            reg = torch.tensor([self.reg]).to(dev)
            ts_id = 0
            for b_id, batch in enumerate(data_loader):
                for each in batch:
                    sa = each.view(-1, 1).to(dev)
                    clbd = lbd[:, data_ids[ts_id]].view(-1, 1)
                    loss = model(sa, self.M, basis, clbd, reg)
                    total_loss += loss.item()
                    ts_id += 1
        return loss / ts_id

    def generateindex(self, output_file='index.tsv', proj_algo='svd',
                      ifscale=False, compare=False):
        "projection algorithm, default 'svd', or 'pca', 'ica'"
        # TODO: generate time-series index from model
        # raise NotImplementedError('generate index')
        with open(self.ckpt, 'rb') as f:
            m, basis, lbd = torch.load(f)

        # load basis and \lambda to cpu
        basis = basis.cpu()
        lbd = lbd.cpu()

        if proj_algo == 'svd':
            proj = TruncatedSVD(n_components=1, random_state=self.seed)
        elif proj_algo == 'ica':
            proj = FastICA(n_components=1, random_state=self.seed)
            raise NotImplementedError('ICA not available')
        elif proj_algo == 'pca':
            proj = PCA(n_components=1, random_state=self.seed)
            raise NotImplementedError('PCA not available')

        basis_proj = torch.tensor(proj.fit_transform(basis.T).T, device='cpu')
        index_docs = basis_proj @ lbd
        ordereddate = OrderedDict(sorted(self.date2idlist.items()))
        interval_len = [len(v) for k, v in ordereddate.items()]

        # pdb.set_trace()
        index = np.array([i.sum()
                          for i in index_docs.flatten().split(interval_len)]).reshape(-1, 1)
        date_interval = np.array(list(ordereddate.keys())).reshape(-1, 1)
        m = np.concatenate([date_interval, index], axis=1)
        df = pd.DataFrame(m, columns=['date', 'index'])
        if ifscale:
            df['index'] = scale(df['index']) + 100
        if not os.path.exists('./results'):
            os.makedirs('./results')
        with open(os.path.join('./results', output_file), 'w') as f:
            df.to_csv(f, sep='\t', header=True, index=False)

        if compare:
            df = pd.read_csv(os.path.join('./results', output_file), sep='\t',
                             index_col='date', parse_dates=True)
            dfcom = pd.read_csv('compare.tsv', sep='\t', index_col='date',
                                parse_dates=True)
            # pdb.set_trace()
            dfcom['index'] = scale(df['index'].loc['1989-01':'2016-08']) + 100
            # praag = np.corrcoef(dfcom['index'], dfcom['indexaag'])[0, 1]
            prnewori = np.corrcoef(dfcom['index'], dfcom['indexori'])[0, 1]
            praagori = np.corrcoef(dfcom['indexaag'], dfcom['indexori'])[0, 1]
            prwigori = np.corrcoef(dfcom['indexwig'], dfcom['indexori'])[0, 1]
            # prwigaag = np.corrcoef(dfcom['indexwig'], dfcom['indexaag'])[0, 1]
            print('Pearson coef ==> ')
            print('AAG: {:.4f}'.format(praagori))
            print('WIG: {:.4f}'.format(prwigori))
            print('New: {:.4f}'.format(prnewori))
            return dfcom

    def idmap(self, dataset, spacy_model, merge_entity, process_fn,
              remove_stop, remove_punct):
        """
        spacy_model: str, spacy language model name
            Default: nlp = spacy.load('en_core_web_sm', disable=["tagger"])
        merge_entity: bool, merge entity detected by spacy model
            Default: True
        process_fn: callable, take (nlp, doc) as input
            Default: None
            Usage: process_fn = my_process_fn(nlp, doc)
            'nlp' is the class imported by spacy lang model
            you can also add your personal pipeline here
        remove_stop: bool, whether to remove stop words
            Default: False
        remove_punct: bool, whether to remove punctuation
            Default: True
        """

        print(f'Loading spacy model {spacy_model}')
        nlp = spacy.load(spacy_model, disable=["tagger"])
        if merge_entity:
            merge_ents = nlp.create_pipe("merge_entities")
            nlp.add_pipe(merge_ents)

        print('preprocessing data with spacy')
        sentences, id2date, id2doc, senid2docid, date2idlist =\
            loaddata_pipe(nlp, dataset, remove_stop,
                          remove_punct, self.interval)
        return sentences, id2date, id2doc, senid2docid, date2idlist


def loaddata_pipe(nlp, dataset, remove_punct, remove_stop, interval):
    def ass(token):
        "return True or False by criterion"
        if remove_punct:
            if remove_stop:  # remove both
                return not (token.is_punct and token.is_stop)
            else:  # remove punct but not stop
                return not token.is_punct
        else:  # remove stop but not punct
            if remove_stop:
                return not token.is_stop
            else:
                return True

    def date2str(date, interval):
        dt = datetime.strptime(date, '%Y-%m-%d')
        if interval == 'M':
            return str(dt.date())[:7]
        elif interval == 'Y':
            return str(dt.date())[:4]
        elif interval == 'D':
            return str(dt.date())
        else:
            raise ValueError("value of 'interval' must be in 'M' 'Y' 'D'")

    dates, docs = zip(*dataset)
    id2date, id2doc, senid2docid = {}, {}, {}
    date2idlist = defaultdict(list)
    sentences = []
    sen_id = 0
    # pdb.set_trace()
    parsed_docs = [[[t.lemma_ for t in sen if ass(t)] for sen in doc.sents]
                   for doc in nlp.pipe(docs, disable=['tagger'])]
    assert len(dates) == len(parsed_docs)

    for id, sens in enumerate(parsed_docs):
        date = dates[id]
        id2date[id] = date
        id2doc[id] = docs[id]
        shortdate = date2str(date, interval)
        sentences += sens
        for sen in sens:
            if len(sen) > 0:
                senid2docid[sen_id] = id
                date2idlist[shortdate].append(sen_id)
                sen_id += 1
    return sentences, id2date, id2doc, senid2docid, date2idlist


def compress_dictionary(gensim_wv, topk=1000, n_clusters=10,
                        l1_reg=0.01, metric='sqeuclidean', seed=0):
    "gensim_wv is gensim_model.wv"
    print('compressing dictionary...')
    vocab = gensim_wv.vocab
    assert topk < len(vocab), 'pick smaller number of topk'
    lasso = linear_model.Lasso(alpha=l1_reg, fit_intercept=False, max_iter=5000)
    base_tokens = []
    # TODO: replace the frequency-based base-tokens
    # for i in range(len(vocab)):
    #     if len(base_tokens) == topk:
    #         break
    #     if gensim_wv.index2entity[i] not in stop_words:
    #         base_tokens.append(gensim_wv.index2entity[i])

    base_tokens = km_compress(gensim_wv, topk, n_clusters, seed)
    # other_tokens = [i for i in vocab.keys() if i not in base_tokens]
    X = torch.tensor(gensim_wv[base_tokens],
                     dtype=dt, device='cpu')  # compressed dist
    M = mjit(dist(X.to(dev), metric=metric))
    # pdb.set_trace()
    id2comdict = []
    for tk in gensim_wv.vocab.keys():
        if tk in base_tokens:
            z = torch.zeros(topk, dtype=dt, device=dev)
            z[base_tokens.index(tk)] == 1.
            id2comdict.append(z)
        else:
            # fit N * k to N * 1, return 1 * k
            lasso.fit(X.T, torch.tensor(gensim_wv[tk],
                                        dtype=dt, device='cpu').view(-1, 1))
            z = torch.tensor(lasso.coef_, dtype=dt, device=dev)
            id2comdict.append(z)
    return M, id2comdict


def km_compress(wv, topk=1000, n_clusters=10, seed=0):
    "default to 10 cluster, each having 100 tokens, total 1000 vocab"
    tk_cluster = round(topk / n_clusters)  # ceil the float

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=-4)
    vl = list(wv.vocab)
    X = wv[wv.vocab]
    km.fit(X)
    k_base = []
    for k in range(n_clusters):
        # get indices for cluster k
        k_indices = np.argwhere(km.labels_ == k).flatten().tolist()
        kidtokens = [(wv.index2word.index(vl[i]), vl[i]) for i in k_indices
                     if vl[i] not in stop_words]
        sort_ktokens = sorted(kidtokens, key=itemgetter(0))
        ct = 0
        for id_tk in sort_ktokens:
            if ct == tk_cluster:
                break
            else:
                k_base.append(id_tk[1])
                ct += 1
        # pdb.set_trace()
    if len(k_base) != topk:
        for i in vl:
            if len(k_base) == topk:
                break
            elif i not in k_base:
                k_base.append(i)

    return k_base


def getfreq_single_compress(gensim_wv, id2comdict, sentence_batch):
    def f(sen):
        senvec = [sen.count(i) for i in gensim_wv.index2word]
        sendist = torch.stack([ct * id2comdict[id]
                               for id, ct in enumerate(senvec)])
        return sendist.sum(dim=0)
    sendist_batch = torch.stack([f(sen) for sen in sentence_batch], dim=0)
    return tensordiv(sendist_batch)


def getfreq_single_original(gensim_wv, sentence_batch):
    """get the word distribution of a certain sentence"""
    def f(sen):
        senvec = [sen.count(i) for i in gensim_wv.index2word]
        sendist = torch.tensor(senvec, dtype=dt, device=dev)
        return sendist
    sendist_batch = torch.stack([f(sen) for sen in sentence_batch], dim=0)
    return tensordiv(sendist_batch)


@torch.jit.script
def tensordiv(sendist: Tensor):
    return sendist / sendist.sum(1).view(-1, 1)


@torch.jit.script
def mjit(M: Tensor):
    return M / M.max()


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
    torch.max(distances, torch.zeros(distances.shape, dtype=dt, device=dev),
              out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances = distances - torch.diag(distances.diag())
    return distances if squared else torch.sqrt(distances, out=distances)
