import os
import pdb
from collections import OrderedDict, defaultdict
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import spacy
import torch
from gensim.models import Word2Vec
from sklearn import linear_model
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from torch import Tensor, nn, optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from model import WasserIndexGen

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
                 train_eval_test=[0.6, 0.1, 0.3],
                 emsize=10,
                 batch_size=64,
                 num_topics=4,
                 reg=0.1,
                 epochs=5,
                 opt='adam',
                 lr=0.005,
                 wdecay=1.2e-6,
                 log_interval=50, seed=0,
                 algorithm='original',
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
                 **kwargs):
        """
        Parameters:
        ======
        dataset         : list, of (date, doc) pairs
        train_eval_test : list, of floats sum to 1, how to split dataset
        emsize          : int, dim of embedding
        batch_size      : int, size of a batch
        num_topics      : int, K topics
        reg             : float, entropic regularization term in Sinkhorn
        epochs          : int, epochs to train
        lr              : float, learning rate for optimizer
        wdecay          : float, L-2 regularization term used by some optimizers
        log_interval    : int, print log one per k steps
        seed            : int, pseudo-random seed for pytorch
        algorithm       : str, 'original' or 'compress'
        opt             : str, which optimizer to use, default to 'adam'
        ckpt_path       : str, checkpoint when training model
        numItermax      : int, max steps to run Sinkhorn, dafault 1000
        dtype           : torch.dtype, default torch.float32
        spacy_model     : str, spacy language model name
                        Default: nlp = spacy.load('en_core_web_sm', disable=["tagger"])
        metric          : str, 'sqeuclidean' or 'euclidean'
        merge_entity    : bool, merge entity detected by spacy model, default True
        process_fn      : callable, arg: (nlp, doc), None to use default
                        Usage: process_fn = my_process_fn(nlp, doc)
                            'nlp' is the class imported by spacy lang model
                            you can also add your personal pipeline here
        remove_stop     : bool, whether to remove stop words, default False
        remove_punct    : bool, whether to remove punctuation, default True
        interval        : 'M', 'Y', 'D'

        Also parameters from Word2Vec
        """
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

        self.log_interval = log_interval
        # self.algorithm = algorithm  # 'original' or 'compressed'

        # ckpt
        self.ckpt = os.path.join(ckpt_path,
                                 f'WIG_Bsz_{batch_size}_K_{num_topics}_LR_{lr}_\
                                 EMsize_{emsize}_Reg_{reg}_Opt_{opt}_Algorithm_{algorithm}')
        self.ckpt_path = ckpt_path
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        # set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        sentences, self.id2date, self.id2doc, self.senid2docid, self.date2idlist = \
            self.idmap(dataset, spacy_model, merge_entity, process_fn,
                       remove_stop, remove_punct)

        # prepare train, eval, and test data
        self.d_l = d_l = len(self.senid2docid.keys())
        assert sum(train_eval_test) == 1., \
            'Three shares do not sum to one.'
        train_r, eval_r, test_r = train_eval_test
        print(f'Splitting data by ratio: {train_r} {eval_r} {test_r}')
        tr_l = round(d_l * train_r)
        ts_l = round(d_l * test_r)
        ev_l = d_l - tr_l - ts_l

        # shuffle ids
        data_ids = torch.randperm(d_l)  # ids of splitted sentences

        # Word2Vec model
        print('run Word2Vec model for word embeddings...')
        word2vec = Word2Vec(sentences=[sentences[i] for i in data_ids],
                            **kwargs)
        wv = word2vec.wv
        self.vocab = wv.vocab

        # choose with algorithm to use, if compressed then shrink vocab dim
        if algorithm == 'original':
            print('algorithm has been chosen as original')
            X = torch.tensor(wv[wv.vocab], dtype=dtype, device=device)
            self.M = mjit(dist(X, metric=metric))
            mycollate = partial(getfreq_single_original, wv)
        elif algorithm == 'compress':
            print('algorithm has been chosen as compressed dictionary')
            self.M, id2comdict = compress_dictionary(wv, topk=1000,
                                                     l1_reg=0.01,
                                                     metric='sqeuclidean')
            mycollate = partial(getfreq_single_compress, wv, id2comdict)
        else:
            # TODO: geodesic regression with l1?
            raise ValueError("use 'original' or 'compress' algotithm")

        tr_ids, ev_ids, ts_ids = data_ids.split([tr_l, ev_l, ts_l])
        self.tr_ids = tr_ids
        self.ev_ids = ev_ids
        self.ts_ids = ts_ids
        self.tr_dl = DataLoader([sentences[i] for i in tr_ids],
                                batch_size=batch_size, collate_fn=mycollate)
        self.ev_dl = DataLoader([sentences[i] for i in ev_ids],
                                batch_size=batch_size, collate_fn=mycollate)
        self.ts_dl = DataLoader([sentences[i] for i in ts_ids],
                                batch_size=batch_size, collate_fn=mycollate)

        global dt
        if dtype:
            dt = dtype

        global dev
        if device == 'cuda':
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            dev = torch.device('cpu')

        self.model = WasserIndexGen(batch_size=batch_size,
                                    num_topics=num_topics,
                                    reg=reg,
                                    numItermax=numItermax,
                                    stopThr=stopThr,
                                    dtype=dt,
                                    device=dev)

        print(f'WIG model {self.model}')

        # init
        self.R = torch.randn((self.M.shape[0], self.num_topics), device=dev)
        self.A = torch.randn((self.num_topics, self.d_l), device=dev)
        self.basis = softmax(self.R, dim=0)  # softmax over columns
        self.lbd = softmax(self.A, dim=0)

    def train(self, train_loader, eval_loader):

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

        total_loss = 0.
        best_loss = 1e9
        for epoch in range(self.epochs):
            tr_id = 0
            for b_id, batch in enumerate(train_loader):
                for sub_id, each in enumerate(batch):
                    csa = each.view(-1, 1).to(dev)
                    clbd = self.lbd[:,
                                    self.tr_ids[tr_id]].view(-1, 1).to(dev)

                    # basis.requires_grad_()
                    # clbd.requires_grad_()
                    reg = torch.tensor([self.reg]).to(dev)

                    loss = self.model(csa, self.M, self.basis, clbd, reg)
                    loss.backward()
                    optimizer.step()
                    # TODO: check learnable parameterss
                    total_loss += loss.item()
                    tr_id += 1

                self.basis = softmax(self.R, dim=0)  # softmax over columns
                self.lbd = softmax(self.A, dim=0)

            # if not self.infer:
            # evaluate after training
            eval_loss = self.evaluate(self.eval_loader, self.ev_ids,
                                      self.basis, self.lbd)
            if eval_loss < best_loss:  # save bast model among all epochs
                with open(self.ckpt, 'wb') as f:
                    torch.save(self.model, f)
                best_loss = eval_loss
            print('*' * 50)
            print(f'Epoch: {epoch}, LR: {self.lr}, \
                    Train Loss: {total_loss/tr_id}, Eval Loss: {eval_loss}')
            print('*' * 50)
            if epoch % self.visualize_every == 0:
                # TODO: implement visualize function per log_interval
                #     visualize(self.model, self.vocab)
                pass
        with open(self.ckpt, 'rb') as f:
            m = torch.load(f)
        eval_loss = self.evaluate(m, self.eval_loader, self.ev_ids,
                                  self.basis, self.lbd)
        print(f'Evaluation Loss: {eval_loss}')
        return eval_loss

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
                sa = batch.view(-1, 1).to(dev)
                clbd = lbd[:, data_ids[ts_id]].view(-1, 1)
                loss = model(sa, self.M, basis, clbd, reg)
                total_loss += loss.item()
                ts_id += 1
        return loss / ts_id

    def generateindex(self, proj_algo='svd'):
        "projection algorithm, default 'svd', or 'pca', 'ica'"
        # TODO: generate time-series index from model
        raise NotImplementedError('generate index')
        with open(self.ckpt, 'rb') as f:
            m = torch.load(f)

        # load basis and \lambda to cpu
        basis = m.basis.cpu()
        lbd = m.lbd.cpu()

        if proj_algo == 'svd':
            proj = TruncatedSVD(n_components=1, random_state=self.seed)
        elif proj_algo == 'ica':
            proj = FastICA(n_components=1, random_state=self.seed)
            raise NotImplementedError('ICA not available')
        elif proj_algo == 'pca':
            proj = PCA(n_components=1, random_state=self.seed)
            raise NotImplementedError('PCA not available')

        basis_proj = proj.fit_transform(basis.T).T
        index_docs = basis_proj @ lbd
        ordereddate = OrderedDict(sorted(self.date2idlist.items()))
        interval_len = [len(v) for k, v in ordereddate.items()]
        # index_docs.split(interval_len)

        index = np.array([i.sum()
                          for i in index_docs.split(interval_len)]).reshape(-1, 1)
        date_interval = np.array(list(ordereddate.keys())).reshape(-1, 1)
        m = np.concatenate([date_interval, index], axis=1)
        df = pd.DataFrame(m, columns=['date', 'index'])
        if not os.path.exists('./results'):
            os.makedirs('./results')
        with open(os.path.join('./results', 'index.tsv'), 'w') as f:
            df.to_csv(f, sep='\t', header=True, index=False)
        pass

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

        def default_fn(nlp, doc):
            spacy_doc = nlp(doc)
            if remove_punct:
                sens = [[i.lemma_ for i in d if not i.is_punct]
                        for d in spacy_doc.sents]
            if remove_stop:
                sens = [[i.lemma_ for i in d if not i.is_stop]
                        for d in spacy_doc.sents]
            return sens
        id2date, id2doc, senid2docid = {}, {}, {}
        date2idlist = defaultdict(list)
        sentences = []
        sen_id = 0
        if merge_entity:
            print(f'Loading spacy model {spacy_model}')
            nlp = spacy.load(spacy_model, disable=["tagger"])
            merge_ents = nlp.create_pipe("merge_entities")
            nlp.add_pipe(merge_ents)
        if not process_fn:  # if not providing
            process = partial(nlp, default_fn)
        else:
            process = partial(nlp, process_fn)
        for id, date_doc in enumerate(dataset):
            date, doc = date_doc
            id2date[id] = date
            id2doc[id] = doc
            shortdate = date2str(date, self.interval)
            date2idlist[shortdate].append(id)
            sens = process(doc)
            sentences += sens
            for i in sens:
                senid2docid[sen_id] = id
                sen_id += 1
        return sentences, id2date, id2doc, senid2docid, date2idlist


def compress_dictionary(gensim_wv, topk=1000, l1_reg=0.01, metric='sqeuclidean'):
    "gensim_wv is gensim_model.wv"
    print('compressing dictionary...')
    vocab = gensim_wv.vocab
    assert topk < len(vocab), 'pick smaller number of topk'
    lasso = linear_model.Lasso(alpha=l1_reg, fit_intercept=False)
    base_tokens = []
    for i in range(len(vocab)):
        if len(base_tokens) == topk:
            break
        if gensim_wv.index2entity[i] not in stop_words:
            base_tokens.append(gensim_wv.index2entity[i])
    # other_tokens = [i for i in vocab.keys() if i not in base_tokens]
    X = torch.tensor(gensim_wv[base_tokens],
                     dtype=dt, device=dev)  # compressed dist
    M = mjit(dist(X, metric=metric))
    id2comdict = []
    for tk in gensim_wv.vocab.keys():
        if tk in base_tokens:
            z = torch.zeros(topk, dtype=dt, device=dev)
            z[base_tokens.index(tk)] == 1.
            id2comdict.append(z)
        else:
            # fit N * k to N * 1, return 1 * k
            lasso.fit(X.T, torch.tensor(gensim_wv[tk],
                                        dtype=dt, device=dev).view(-1, 1))
            z = torch.tensor(lasso.coef_, dtype=dt, device=dev)
            id2comdict.append(z)
    # return base_tokens, other_tokens
    return M, id2comdict


def getfreq_single_compress(gensim_wv, id2comdict, sentence):
    senvec = [sentence.count(i) for i in gensim_wv.index2word]
    sendist = torch.stack([ct * id2comdict[id] for id, ct in enumerate(senvec)])
    return tensordiv(sendist.sum(dim=0))


def getfreq_single_original(gensim_wv, sentence):
    """get the word distribution of a certain sentence"""
    # assert sentence != []
    senvec = [sentence.count(i) for i in gensim_wv.index2word]
    sendist = torch.tensor(senvec, dtype=dt, device=dev)
    # pdb.set_trace()
    return tensordiv(sendist)


@torch.jit.script
def tensordiv(sendist: Tensor):
    return torch.div(sendist, sendist.sum())


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
