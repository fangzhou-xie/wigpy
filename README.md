wigpy: Computing Wasserstein Index Generation model and Pruned WIG
================
Fangzhou Xie (fangzhou\[dot\]xie\[at\]nyu\[dot\]edu)
February 29, 2020

wigpy
-----

This `wigpy` package is to compute time-series indices from texts, using Wasserstein Index Generation (WIG) model and pruned Wasserstein Index Generation (pWIG) model. The former is described in *Wasserstein Index Generation Model: Automatic generation of time-series index with application to Economic Policy Uncertainty* ([published version](https://www.sciencedirect.com/science/article/pii/S0165176519304410?via%3Dihub) and [arxiv](https://arxiv.org/abs/1908.04369)); while the latter is described in ([ssrn and under review](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3546407)) as an extension to the original WIG model to deal with large-vocabulary instances.

Dependencies
------------

-   python 3.7.6
-   pytorch 1.3.1
-   scikit-learn 0.22.1
-   numpy 1.18.1
-   pandas 0.25.3
-   spacy 2.2.3
-   gensim 3.8.1

Note: This package is developed under Ubuntu 18.04.3 LTS but not tested on macOS or Windows machines. I think macOS should work fine but I highly doubt it is the case for Windows users. I only listed the version of packages that I am using, and other (previous) versions may also work as well.

Installation
------------

Change to your project directory and clone this repository:

    cd path/to/your/project
    git clone https://github.com/mark-fangzhou-xie/wigpy.git

Usage
-----

### How to load data in to WIG class

The main model class is `WIG` and simply pass your time-attached sentences to it.

    from wigpy.wig import WIG

    sentences = [('2018-01-01', 'This is the first sentence.'),
                 ('2020-02-14', 'I have another sentence.')]

    wig = WIG(sentences)

Note that the input `sentences` here is a list of `(date, sentence)` pair (time-associated texts are required for the WIG model to generate time-series indices). Each time-stamp should be in format "%Y-%m-%d" and make sure to transform your date into this format before calling the `WIG` class.

You could well have multiple sentences for one time-label. In other words, it is fine to pass a whole document for the "date-doc" pair, like:

    sentences = [('2018-01-01', 'I have one sentence here. And another.'),
                 ('2020-02-14', 'There is something else.')]

Under the hood, the parsing will be carried out by `spacy` and to be preferred to use GPU for acceleration. If GPU(CUDA) is not available, then it will fall back to use CPU. In the data processing pipeline, the full document will be parsed into sentences, and you can choose `remove_stop` and/or `remove_punct` to remove stop words and/or punctuation marks in the sentences. <!-- For usage of spacy, please refer to
(https://spacy.io/)[https://spacy.io/]. -->

What is more, you could also pass documents that have same time-stamps. Example:

    sentences = [('2020-02-14', 'A sentence for this date.'),
                 ('2020-02-14', 'Another sentence for today.')]

This also works so you don't have to merge documents beforehand by yourself. In many applications, the texts are observed several times a day and you can pack them all in a list to the `WIG` and it will take care of everything else.

### Model Parameters

As this package is a new implementation of WIG model so the computation is slightly different than the results in the original WIG paper. But the default value are mainly the ones chosen by cross-validation as in the paper.

    Parameters:
    ======
    dataset         : list, of (date, doc) pairs
    train_test_ratio: list, of floats sum to 1, how to split dataset
    emsize          : int, dimension of word embedding (Word2Vec)
    batch_size      : int, size of a batch
    num_topics      : int, K topics to cluster
    reg             : float, entropic regularization term in Sinkhorn
    epochs          : int, epochs to train
    lr              : float, learning rate for optimizer
    wdecay          : float, L-2 regularization term used by some optimizers
    log_interval    : int, print log one per k steps
    seed            : int, pseudo-random seed for pytorch
    prune_topk      : int, max no of tokens to use for pruning vocabulary
    l1_reg          : float, L1 penalty for pruning
    n_clusters      : int, KMeans clusters to group tokens
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
    visualize_every : int, not implemented
    loss_per_batch  : bool, if print loss per batch
    **kwargs        : Also parameters from Word2Vec

The hyperparameters should be self-evident. Any parameter for `gensim.models.Word2Vec` ([Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)) may also be passed into the WIG model. Only remember it is equivalent to pass `emsize` or `size` for the embedding dimension, as the latter is defined by Word2Vec model.

For optimizers, the default is `Adam` as is used by the original WIG algorithm. There are several others to choose:

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

Some of the optimizers have `weight_decay` parameter to have L-2 regularization during optimization step.

For preprocessing, it is performed by `spacy` and there are 4 parameters to control its behavior. `spacy_model` is the language model used by `spacy`, when dealing with other languages or if you want to use large version of English model, be sure to pass it in by this argument. `merge_entity` is a Boolean value and `spacy` will recognize entities and merge them as a single token, e.g. 'quantum mechanics'. The default value is true and you could turn it off by 'False'. `remove_stop` and `remove_punct` are Boolean values to remove (or not) stop words and (or) punctuations. The default behavior is to remove punctuation but not stop words.

### Two Algorithms (Original WIG and pruned WIG)

To use the original algorithm, pass `prune_topk=0` to the `WIG` model, or choose any integer larger than 0 as the dimension of vocabulary, e.g. `prune_topk=1000` for choosing 1000 words as maximum vocabulary length. Those, in this case 1000 words, are called "base tokens", and other words will be approximated by those "base tokens".

The original WIG algorithm relies largely on Wasserstein Dictionary Learning ([here](https://arxiv.org/abs/1708.01955)), as WIG use WDL to cluster documents and then use SVD to produce uni-dimensional time-series index. I implemented the code in python with pytorch calculating gradients, and readers interested in details should refer to the WIG paper.

However, this model leverages Wasserstein distance, which is notoriously expensive for computation. Even with the Sinkhorn regularization, the Optimal-Transport based calculations are still slow to compute. Further, the WIG model requires a full ![N \\times N](https://latex.codecogs.com/png.latex?N%20%5Ctimes%20N "N \times N") matrix to be calculated, where ![N](https://latex.codecogs.com/png.latex?N "N") is the dimension of vocabulary. It is obvious that when the dataset and vocabulary is large, the memory become an issue, especially if we still want to use GPU for acceleration (VRAMs are typically smaller than RAMs). Thus, I propose this modified version--compressed dictionary WIG to shrink vocabulary to a smaller dimension.

First we need to identify the "base tokens". The idea is to choose a subset of vocabulary, of length ![B](https://latex.codecogs.com/png.latex?B "B"), to represent the whole vocabulary, of length ![N](https://latex.codecogs.com/png.latex?N "N"), where ![B &lt;&lt; N](https://latex.codecogs.com/png.latex?B%20%3C%3C%20N "B << N"). Now the question become: how should we choose the "base tokens" wisely? I first consider the unsupervised ![k](https://latex.codecogs.com/png.latex?k "k")-means clustering on word vectors (given by Word2Vec) to first "word-clusters" and pick the most frequent tokens among each clusters.

Formally speaking, we set up the following minimization problem:

![\\argmin\_{\\mathcal{K}\_1,\\cdots,\\mathcal{K}\_n}\\sum\_{k=1}^{K}\\sum\_{x\\in \\mathcal{K}\_k}||x- \\mu\_k||,](https://latex.codecogs.com/png.latex?%5Cargmin_%7B%5Cmathcal%7BK%7D_1%2C%5Ccdots%2C%5Cmathcal%7BK%7D_n%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Csum_%7Bx%5Cin%20%5Cmathcal%7BK%7D_k%7D%7C%7Cx-%20%5Cmu_k%7C%7C%2C "\argmin_{\mathcal{K}_1,\cdots,\mathcal{K}_n}\sum_{k=1}^{K}\sum_{x\in \mathcal{K}_k}||x- \mu_k||,")

 and we could find ![K](https://latex.codecogs.com/png.latex?K "K") clusters of words.

The number of tokens in a cluster is given by the formula: ![\\frac{B}{K}](https://latex.codecogs.com/png.latex?%5Cfrac%7BB%7D%7BK%7D "\frac{B}{K}"), where ![B](https://latex.codecogs.com/png.latex?B "B") is length of "base tokens" and ![K](https://latex.codecogs.com/png.latex?K "K") is the number of clusters. In this way, we are picking tokens from clusters equally.

Now that we have the "base tokens" at hand, the next step is to represent the whole vocabulary by these "base tokens". This step is conducted by using LASSO regression. Denote word vector of base tokens as ![v\_b](https://latex.codecogs.com/png.latex?v_b "v_b") and other tokens as ![v\_o](https://latex.codecogs.com/png.latex?v_o "v_o"), we have

![v\_o = \\sum\_{b=1}^{B}\\alpha\_{o,b}v\_b + \\lambda\\sum\_{b=1}^{B}|\\alpha\_{o,b}|,](https://latex.codecogs.com/png.latex?v_o%20%3D%20%5Csum_%7Bb%3D1%7D%5E%7BB%7D%5Calpha_%7Bo%2Cb%7Dv_b%20%2B%20%5Clambda%5Csum_%7Bb%3D1%7D%5E%7BB%7D%7C%5Calpha_%7Bo%2Cb%7D%7C%2C "v_o = \sum_{b=1}^{B}\alpha_{o,b}v_b + \lambda\sum_{b=1}^{B}|\alpha_{o,b}|,")

and for each ![o](https://latex.codecogs.com/png.latex?o "o"), we have a weight vector of length ![B](https://latex.codecogs.com/png.latex?B "B") to represent token ![o](https://latex.codecogs.com/png.latex?o "o") in the ![B](https://latex.codecogs.com/png.latex?B "B")-dimensional space. The original WIG model will calculate ![N](https://latex.codecogs.com/png.latex?N "N")-dimensional word frequency vector for the further transportation computation, and now we need to use the ![\\alpha\_o = \[\\alpha\_{o,b}\]](https://latex.codecogs.com/png.latex?%5Calpha_o%20%3D%20%5B%5Calpha_%7Bo%2Cb%7D%5D "\alpha_o = [\alpha_{o,b}]") to represent this ![o](https://latex.codecogs.com/png.latex?o "o") token. Finally, we could compute pair-wise word-distance matrix ![B\\times B](https://latex.codecogs.com/png.latex?B%5Ctimes%20B "B\times B") for the Sinkhorn computation.
