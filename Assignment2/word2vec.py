from __future__ import print_function
import numpy as np
from collections import Counter
from scipy.special import expit as sigmoid
from utils import *
import random, time

np.random.seed(1234)

# P(+ | t, c)
def compute_pos_prob(t_v, c_v):
    return sigmoid(np.dot(t_v,c_v))


# P(- | t, n_i)
def compute_neg_prob(t_v, n_v):
    return sigmoid(-np.dot(t_v,n_v))


# compute the SGNS loss and the partial derivatives of the loss
# with respect to the word/context embeddings
def compute_obj_and_grad(t, c, W, C, sampling_dist, do_grad_check=False):
    t_v = W[t] # the target word vector
    c_v = C[c] # the context word vector
    ns = negative_sampling(sampling_dist, do_grad_check) # get some negative samples

    dW = np.zeros(W.shape) # this matrix contains target word derivatives (dL/dW)
    dC = np.zeros(C.shape) # this matrix contains context word derivatives (dL/dC)
    L = 0. # the SGNS objective is stored in this variable

    # computes P(+ | t, c) and adds log(P) to objective
    pos_prob = compute_pos_prob(t_v, c_v)
    L += np.log(pos_prob)

    # compute derivatives of the log P(+ | t, c) term WRT target and context word
    dW[t] += (1-pos_prob)*c_v # IMPLEMENT ME!
    dC[c] += (1-pos_prob)*t_v # IMPLEMENT ME!

    # then compute P(- | t, n_i) for each negative sample and add to objective
    for n_i in ns:
        n_v = C[n_i]
        neg_prob = compute_neg_prob(t_v, n_v)
        L += np.log(neg_prob)

        # compute derivative of the log P(- | t, n_i) term WRT target and context word 
        dW[t] += -(1-neg_prob)*n_v # IMPLEMENT ME!
        dC[n_i] += -(1-neg_prob)*t_v # IMPLEMENT ME!

    return L, dW, dC


# loop through each context in the data, make updates using SGNS
def train(idxes, window_size, num_epochs, sampling_dist, W, C, \
    vocab, idx_to_w, learning_rate=0.01):

    for ep in range(num_epochs):
        L = 0. # cumulative objective function
        num_ctxs = 0.
        start_time = time.time()

        for subseq in next_context(idxes, window_size):

            # use each word as a target word, all other words as contexts
            for idx, target_word in enumerate(subseq):
                context = [subseq[i] for i in range(window_size) if i != idx]
                
                # for each target word / context word pair, compute the SGNS objective
                for context_word in context:
                    ex_L, dW, dC = compute_obj_and_grad(target_word, context_word,\
                        W, C, sampling_dist)
                    L += ex_L # add this example's contribution to objective
                    W += learning_rate * dW # update W
                    C += learning_rate * dC # update C

                num_ctxs += 1

        print(ep, L / num_ctxs, time.time() - start_time)


if __name__ == '__main__':

    # read in the small data file
    data_file = 'small_text8' # YOUR PATH HERE
    num_epochs = 3 # number of passes through the dataset to make
    text, vocab, sampling_dist = compute_vocab(data_file) 
    idx_to_w = dict((v,k) for (k,v) in vocab.iteritems()) # useful for eval / debugging

    # onto some real-world model hyperparameters...
    dim = 25 # dimensionality of word vectors
    window_size = 4 # how many total surrounding words to include in context
    vocab_size = len(vocab) # number of word types in vocab

    W, C = init_parameters(dim, vocab_size) # new embeddings
    print(W.shape, C.shape, len(text))

    train(text, window_size, num_epochs, sampling_dist, W, C, \
        vocab, idx_to_w, learning_rate=1.0)





