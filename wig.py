import pdb
from functools import partial

import spacy
import torch
from gensim.models import Word2Vec
from sklearn import linear_model
from torch import Tensor, nn, optim
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
                 batch_size=64,
                 num_topics=4,
                 reg=0.1,
                 epochs=5,
                 lr=0.005,
                 wdecay=1.2e-6,
                 bow_norm=False,
                 w2v_paradict={'min_count': 1,
                               'size': 10},
                 log_interval=50, seed=0,
                 algorithm='original',
                 opt='adam',
                 ckpt_path='./ckpt',
                 numItermax=1000,
                 stopThr=1e-9,
                 dtype=torch.float32,
                 spacy_model='en_core_web_sm',
                 metric='sqeuclidean',
                 merge_entity=True,
                 process_fn=None,
                 remove_stop=False,
                 remove_punct=True):
        """
        Parameters:
        ======
        dataset         : list, of (date, doc) pairs
        train_eval_test : list, of floats sum to 1, how to split dataset
        batch_size      : int, size of a batch
        num_topics      : int, K topics
        reg             : float, entropic regularization term in Sinkhorn
        epochs          : int, epochs to train
        w2v_paradict    : dict, parameters of gensim.models.Word2Vecs.
                          size => emsize
        lr              : float, learning rate for optimizer
        wdecay          : float, L-2 regularization term used by some optimizers
        bow_norm        : bool, normalize each batch before feed into model
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
        """

        # set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        sentences, self.id2date, self.id2doc, self.senid2docid = \
            self.idmap(dataset, spacy_model, merge_entity, process_fn,
                       remove_stop, remove_punct)

        # prepare train, eval, and test data
        d_l = len(self.senid2docid.keys())
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
        word2vec = Word2Vec(sentences=[sentences[i] for i in data_ids],
                            **w2v_paradict)
        wv = word2vec.wv

        if algorithm == 'original':
            X = torch.tensor(wv[wv.vocab], dtype=dtype, device=device)
            M = mjit(dist(X, metric=metric))
            mycollate = partial(getfreq_single_original, wv)
        elif algorithm == 'compress':
            M, id2comdict = compress_dictionary(wv,
                                                topk=1000,
                                                l1_reg=0.01,
                                                metric='sqeuclidean')
            mycollate = partial(getfreq_single_compress, wv, id2comdict)
        else:
            raise ValueError("use 'original' or 'compress' algotithm")

        tr_ids, ev_ids, ts_ids = data_ids.split([tr_l, ev_l, ts_l])
        self.tr_dl = DataLoader([sentences[i] for i in tr_ids],
                                batch_size=batch_size, collate_fn=mycollate)
        self.ev_dl = DataLoader([sentences[i] for i in ev_ids],
                                batch_size=batch_size, collate_fn=mycollate)
        self.ts_dl = DataLoader([sentences[i] for i in ts_ids],
                                batch_size=batch_size, collate_fn=mycollate)

        self.lr = lr
        self.epochs = epochs
        self.log_interval = log_interval
        self.algorithm = algorithm  # 'original' or 'compressed'

        if dtype:
            global dtype
            dtype = dtype

        if device == 'cuda':
            global device
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            global device
            device = torch.device('cpu')

        self.model = WasserIndexGen(batch_size=batch_size,
                                    num_topics=num_topics,
                                    reg=reg,
                                    numItermax=numItermax,
                                    stopThr=stopThr,
                                    dtype=dtype,
                                    device=device)

        print(f'WIG model {self.model}')
        if opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                        weight_decay=wdecay)
        elif opt == 'adagrad':
            self.optimizer = optim.Adagrad(
                self.model.parameters(), lr=lr, weight_decay=wdecay)
        elif opt == 'adadelta':
            self.optimizer = optim.Adadelta(
                self.model.parameters(), lr=lr, weight_decay=wdecay)
        elif opt == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, weight_decay=wdecay)
        elif opt == 'asgd':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=lr,
                                        t0=0, lambd=0., weight_decay=wdecay)
        else:
            print('Optimizer not supported . Defaulting to vanilla SGD...')
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

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
            sens = process(doc)
            sentences += sens
            for i in sens:
                senid2docid[sen_id] = id
                sen_id += 1
        return sentences, id2date, id2doc, senid2docid

    def train(self, train_loader, eval_loader):
        pass

    def evaluate(self, eval_loader, test_loader):
        pass

    def generateindex(self, time_text):
        pass


def compress_dictionary(gensim_wv, topk=1000, l1_reg=0.01, metric='sqeuclidean'):
    "gensim_wv is gensim_model.wv"
    vocab = gensim_wv.vocab
    assert topk < len(vocab), \
        'topk excess dictionary length, pick smaller number'
    lasso = linear_model.Lasso(alpha=l1_reg, fit_intercept=False)
    base_tokens = []
    for i in range(len(vocab)):
        if len(base_tokens) == topk:
            break
        if gensim_wv.index2entity[i] not in stop_words:
            base_tokens.append(gensim_wv.index2entity[i])
    # other_tokens = [i for i in vocab.keys() if i not in base_tokens]
    X = torch.tensor(gensim_wv[base_tokens],
                     dtype=dtype,
                     device=device)  # compressed dist
    M = mjit(dist(X, metric=metric))
    id2comdict = []
    for tk in gensim_wv.vocab.keys():
        if tk in base_tokens:
            z = torch.zeros(topk, dtype=dtype, device=device)
            z[base_tokens.index(tk)] == 1.
            id2comdict.append(z)
        else:
            # fit N * k to N * 1, return 1 * k
            lasso.fit(X.T, torch.tensor(gensim_wv[tk],
                                        dtype=dtype, device=device).view(-1, 1))
            z = torch.tensor(lasso.coef_, dtype=dtype, device=device)
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
    sendist = torch.tensor(senvec, dtype=dtype, device=device)
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
    torch.max(distances, torch.zeros(distances.shape, dtype=dtype,
                                     device=device), out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances = distances - torch.diag(distances.diag())
    return distances if squared else torch.sqrt(distances, out=distances)


class Lasso(nn.Module):
    "Lasso for compressing dictionary, xxxxxx"

    def __init__(self, input_size):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def lasso(x, y, lr=0.005, max_iter=2000, tol=1e-4, opt='sgd'):
    # x = x.detach()
    # y = y.detach()
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='sum')
    if opt == 'adam':
        optimizer = optim.Adam(lso.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(lso.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(lso.parameters(), lr=lr)
    w_prev = torch.tensor(0.)
    for it in range(max_iter):
        # lso.linear.zero_grad()
        optimizer.zero_grad()
        out = lso(x)
        loss = criterion(out, y)
        l1_norm = 0.1 * torch.norm(lso.linear.weight, p=1)
        loss += l1_norm
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        w = lso.linear.weight.detach()
        if bool(torch.norm(w_prev - w) < tol):
            break
        w_prev = w
        # if it % 100 == 0:
        # print(loss.item() - loss_prev)
    return lso.linear.weight.detach()


# a = torch.randn(4, 5)
# b = torch.randn(4, 1)
#
# r = lasso(a, b, opt='adam')
# print(r)
# l = linear_model.Lasso(alpha=0.1, fit_intercept=False)
# l.fit(a, b)
# # l.path(a, b, verbose=True)
# print(l.coef_)
