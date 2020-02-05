import pdb

import torch
from sklearn import linear_model
from torch import Tensor, nn, optim

from model import WasserIndexGen

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, emsize=10, batch_size=64, num_topics=4, reg=0.1,
                 epochs=5,  lr=0.005, wdecay=1.2e-6,
                 log_interval=50, seed=0,
                 algorithm='original', opt='sgd', bow_norm=False,
                 ckpt_path='./ckpt',
                 numItermax=1000, stopThr=1e-9, dtype=torch.float32):

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

        # set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.model = WasserIndexGen(emsize=emsize,
                                    batch_size=batch_size,
                                    num_topics=num_topics,
                                    reg=reg,
                                    numItermax=numItermax,
                                    stopThr=stopThr,
                                    dtype=dtype)

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
            print('Defaulting to vanilla SGD')
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def train(self, train_data, evel_data):
        pass

    def evaluate(self, eval_data, test_data):
        pass


def compress_dictionary(gensim_wv, topk=1000, l1_reg=0.01, metric='sqeuclidean',
                        dtype=torch.float32):
    "gensim_wv is gensim_model.wv"
    vocab = gensim_wv.vocab
    assert topk < len(vocab), \
        'topk excess dictionary length, pick smaller number'
    # l = linear_model.Lasso(alpha=l1_reg, fit_intercept=False)
    base_tokens = []
    for i in range(len(vocab)):
        if len(base_tokens) == topk:
            break
        if gensim_wv.index2entity[i] not in stop_words:
            base_tokens.append(gensim_wv.index2entity[i])
    other_tokens = [i for i in vocab.keys() if i not in base_tokens]
    # X = torch.tensor(gensim_wv[base_tokens], dtype=dtype)
    # M = mjit(dist(X, metric=metric))
    return base_tokens, other_tokens


def getfreq_single_compress(wv_vocab, base_tokens, other_tokens, l1_reg,
                            sentence):
    pass


def getfreq_single_original(wv_vocab, sentence):
    """get the word distribution of a certain sentence"""
    # assert sentence != []
    senvec = [sentence.count(i) for i in wv_vocab.index2word]
    sendist = torch.tensor(senvec, dtype=torch.float32)
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
    torch.max(distances, torch.zeros(
        distances.shape, device=device), out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances = distances - torch.diag(distances.diag())
    return distances if squared else torch.sqrt(distances, out=distances)


class Lasso(nn.Module):
    "Lasso for compressing dictionary"

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
