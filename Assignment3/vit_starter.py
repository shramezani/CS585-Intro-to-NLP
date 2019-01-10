from __future__ import division

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k].
    Behavior undefined if ties (python docs might give clues)"""
    return max(dct.keys(), key=lambda k: dct[k])

def goodness_score(seq, A, B):

    
    #seq = POS sequence
    #B[t][seq[t]] = probability of tth words being seq[t]
    # Output no score for a sequence of none
    for x in seq:
        if x == None:
            return "No score"
        
    # the total "goodness" score of the proposed sequence
    N = len(B)
    score = 0
    score += sum(A[seq[t],seq[t+1]] for t in range(N-1))
    score += sum(B[t][seq[t]] for t in range(N))
    return score

def exhaustive(A, B, output_vocab):
    # the exhaustive decoding algorithm.
    N = len(B)  # length of entire sentence

    def allpaths(sofar):
        # Recursively generate all sequences given a prefix "sofar".
        # this probably could be redone cleverly as a python generator
        retpaths = []
        if len(sofar)==N:
            return [sofar]
        for sym in output_vocab:
            newpath = sofar[:] + [sym]
            retpaths += allpaths(newpath)
        return retpaths

    path_scores = {}
    for path in allpaths([]):
        path = tuple(path)  # tuple version can be used as dict key
        score = goodness_score(path, A, B)
        path_scores[path] = score
    bestseq = dict_argmax(path_scores)
    return list(bestseq)  # might as well convert it to a list, why not

def viterbi(A, B, output_vocab):
    """
    A: a dict of key:value pairs of the form
        {(curtag,nexttag): score}
    with keys for all K^2 possible neighboring combinations,
    and scores are numbers.  We assume they should be used ADDITIVELY, i.e. in log space.
    higher scores mean MORE PREFERRED by the model.

    B: a list where each entry is a dict {tag:score}, so like
    [ {Noun:-1.2, Adj:-3.4}, {Noun:-0.2, Adj:-7.1}, .... ]
    each entry in the list corresponds to each position in the input.

    output_vocab: a set of strings, which is the vocabulary of possible output
    symbols.

    RETURNS:
    the tag sequence yvec with the highest goodness score
    """

    N = len(B)   # length of input sentence

    # viterbi log-prob tables
    V = [{tag:None for tag in output_vocab} for t in range(N)]
    #print("v:",V)
    # backpointer tables
    # back[0] could be left empty. it will never be used.
    back = [{tag:None for tag in output_vocab} for t in range(N)]

    # todo implement the main viterbi loop here
    # you may want to handle the t=0 case separately
    if N==0:
        return[]
    # Memoization
    ## baseline  
    path =[]
    for i in V[0]:
        V[0][i]=B[0][i]
    
    for t in range(1,len(V)):
        for i in V[t]:
            max_=0
            max_index=-1
            for j in V[t-1]:
                if(V[t-1][j]+A[j,i]>max_):
                    max_=V[t-1][j]+A[j,i]
                    max_index=j
            V[t][i] = V[t-1][max_index]+A[max_index,i]+B[t][i]
            back[t][i] = max_index
  


    path.append(dict_argmax(V[N-1]))
    for i in range( len(back)-1):
        path.append(back[len(back)-i-1][path[i]])
        
    
    #print("V: ",dict_argmax(V[N-1]))
    #print("back: ",path)
    # todo implement backtrace also
    return list(reversed(path))

def randomized_test(N=3, V=5):
    # This creates a random model and checks if the exhaustive and viterbi
    # decoders agree.
    import random
    A = { (a,b): random.random() for a in range(V) for b in range(V) }
    Bs = [ [random.random() for k in range(V)] for i in range(N)]

    #print("output_vocab=", range(V))
    #print("A=",A)
    #print("Bs=",Bs)
    

    fromex  = exhaustive(A,Bs, range(V))
    fromvit = viterbi(A,Bs,range(V))
    print(goodness_score(fromex, A, Bs))
    print(fromex)
    print(goodness_score(fromvit, A, Bs))
    print(fromvit)
    assert fromex==fromvit
    print("Worked!")

if __name__=='__main__':
    
    A = {(0,0):2, (0,1):1, (1,0):0, (1,1):5}
    Bs= [ [0,1], [0,1], [25,0] ]
    # that's equivalent to: [ {0:0,1:1}, {0:0,1:1}, {0:25,1:0} ]

    y = exhaustive(A, Bs, set([0,1]))
    print("Exhaustive decoding:", y)
    print("score:", goodness_score(y, A, Bs))
    y = viterbi(A, Bs, set([0,1]))
    print("Viterbi    decoding:", y)
