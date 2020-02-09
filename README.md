# wigpy

This `wigpy` package is to compute time-series indices from texts,
using Wasserstein Index Generation (WIG) model,
as described in
*Wasserstein Index Generation Model:
Automatic generation of time-series
index with application to Economic Policy Uncertainty* by
Fangzhou Xie.
([click here](https://www.sciencedirect.com/science/article/pii/S0165176519304410?via%3Dihub)).


I further propose a variant to ease its computational burden,
compressed dictionary WIG (cWIG). This version is also included.

## Dependencies
* python 3.7.6
* pytorch 1.3.1
* scikit-learn 0.22.1
* numpy 1.18.1
* pandas 0.25.3
* spacy 2.2.3
* gensim 3.8.1

Note: This package is developed under Ubuntu 18.04.3 LTS
but not tested on macOS or Windows machines.
I think macOS should work fine but I highly doubt it is the case
for Windows users.
I only listed the version of packages that I am
using, and other (previous) versions may also work as well.

## Installation

Change to your project directory and clone this repository:

```
cd path/to/your/project
git clone https://github.com/mark-fangzhou-xie/wigpy.git
```


## Usage


### How to load data in to WIG class

The main model class is `WIG` and simply pass your sentences to it.


```
from wigpy.wig import WIG

sentences = [('2018-01-01', 'This is the first sentence.'),
             ('2020-02-14', 'I have another sentence.')]

wig = WIG(sentences)
```


Note that the input `sentences` here is a list of
`(date, sentence)` pair
(time-associated texts are required for the WIG model to generate
time-series indices).
Each time-stamp should be in format "%Y-%m-%d"
and make sure to transform your date into this format before
calling the `WIG` class.

You could well have multiple sentences for one time-label.
In other words, it is fine to pass a whole document for the "date-doc"
pair, like:
```
sentences = [('2018-01-01', 'I have one sentence here. And another.'),
             ('2020-02-14', 'There is something else.')]
```

Under the hood, the parsing will be carried out by `spacy` and
to be preferred to use GPU for acceleration. If GPU(CUDA) is not
available, then it will fall back to use CPU.
In the data processing pipeline,
the full document will be parsed into sentences,
and you can choose `remove_stop` and `remove_punct`
to remove stop words and (or) punctuation marks in the
sentences.
<!-- For usage of spacy, please refer to
(https://spacy.io/)[https://spacy.io/]. -->

What is more, you could also pass documents that have same time-stamps.
Example:

```
sentences = [('2020-02-14', 'A sentence for this date.'),
             ('2020-02-14', 'Another sentence for today.')]

```

This also works so you don't have to merge documents beforehand
by yourself. In many applications, the texts are observed
several times a day and you can pack them all in a list to
the `WIG` and it will take care of everything else.


### Two Algorithms (Original WIG and Compressed Dictionary WIG)

To use the original algorithm, pass `compress_topk=0` to the `WIG`
model, or choose any integer larger than 0 as the dimension
of vocabulary, e.g. `compress_topk=1000` for choosing
1000 words as maximum vocabulary length.
Those, in this case 1000 words, are called "base tokens",
and other words will be approximated by those "base tokens".


The original WIG algorithm relies largely on
Wasserstein Dictionary Learning
([here](https://arxiv.org/abs/1708.01955)),
as WIG use WDL to cluster documents and then use SVD to produce
uni-dimensional time-series index.
I implemented the code in python with pytorch calculating gradients.

However, this model leverages Wasserstein distance,
which is notoriously expensive for computation.
It requires a full
![equation](https://latex.codecogs.com/gif.latex?N%20%5Ctimes%20N)
matrix to be calculated, where
![equation](https://latex.codecogs.com/gif.latex?N)
is the dimension of vocabulary.
It is obvious that when the dataset and vocabulary
is large, the memory become an issue, especially if we still want
to use GPU for acceleration. Thus, I propose this
modified version--compressed dictionary WIG to shrink vocabulary
to a smaller dimension.

First we need to identify the "base tokens". This step
is performed to choose the most frequent
![equation](https://latex.codecogs.com/gif.latex?B) words
that are not "stop words".
If you choose not to remove stop words from the corpus
but only punctuation marks, as is the default setting,
stop words will be kept in the vocabulary.
As they are not informative, we certainly don't want
them to be in our "base tokens".
We further wish choose the most frequently-appeared tokens,
whose length is determined by the `compress_topk` parameter
in the `WIG` model.

Then we need to represent the soon-to-be-removed vocabulary
by the "base tokens".
Notice that "tokens" here have been passed to `gensim` Word2Vec
model to get each of their vector representation.
I choose Lasso regression to find weights for each
of the "non-important" words,

![equation](https://latex.codecogs.com/gif.latex?v_o%20%3D%20%5Csum_%7Bi%3D1%7D%5EB%20%5Calpha_i%20v_i%20&plus;%20%5Clambda%5Csum_%7Bi%3D1%7D%5EB%7C%5Calpha_i%7C),

where ![equation](https://latex.codecogs.com/gif.latex?v_o)
is the word vector of those which will be represented and pruned
("other words"),
and ![equation](https://latex.codecogs.com/gif.latex?v_i)
are the word vectors for "base tokens".
We perform this Lasso regression for all
![equation](https://latex.codecogs.com/gif.latex?o)
such that
![equation](https://latex.codecogs.com/gif.latex?v_o) is not chosen
as "base tokens".

Thus, we only need to calculate the
![equation](https://latex.codecogs.com/gif.latex?B%20%5Ctimes%20B)
matrix (since other tokens will be represented as those)
for the Sinkhorn computation.

### Model Parameters


```
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
compress_topk   : int, max no of tokens to use for compressed algo
opt             : str, which optimizer to use, default to 'adam'
ckpt_path       : str, checkpoint when training model
numItermax      : int, max steps to run Sinkhorn, dafault 1000
dtype           : torch.dtype, default torch.float32
spacy_model     : str, spacy language model name
                Default: nlp = spacy.load('en_core_web_sm', disable=["tagger"])
metric          : str, 'sqeuclidean' or 'euclidean'
merge_entity    : bool, merge entity detected by spacy model, default True
remove_stop     : bool, whether to remove stop words, default False
remove_punct    : bool, whether to remove punctuation, default True
interval        : 'M', 'Y', 'D'
visualize_every : int,
loss_per_batch  : bool, if print loss per batch

Also parameters from Word2Vec (gensim)
```

As this package is a new implementation of WIG model
so the computation is slightly different
than the results in the original WIG paper.

For optimizers, the default is `Adam` as is used by the original
WIG algorithm. There are several others to choose:
```
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
```
