from __future__ import division

import matplotlib.pyplot as plt
import math
import os
import time
import operator

import json, ast

from collections import defaultdict, Counter


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'


###### DO NOT MODIFY THIS FUNCTION #####
def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)
###### END FUNCTION #####


def n_word_types(word_counts):
    '''
    return a count of all word types in the corpus
    using information from word_counts
    '''
    return len(word_counts)


def n_word_tokens(word_counts):
    '''
    return a count of all word tokens in the corpus
    using information from word_counts
    '''
    return int(sum(word_counts.values()))



class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, path_to_data, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.train_dir = os.path.join(path_to_data, "train")
        self.test_dir = os.path.join(path_to_data, "test")
        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }
        
    def find_first_misclassified(self, alpha):
        """
        DO NOT MODIFY THIS FUNCTION

        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        pos_path = os.path.join(self.test_dir, POS_LABEL)
        neg_path = os.path.join(self.test_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read().decode('utf8')
                    bow = self.tokenize_doc(content)
                    if self.classify(bow, alpha) != label:
                        return content, self.classify(bow, alpha)
        return ""
    
    
    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
         """

        pos_path = os.path.join(self.train_dir, POS_LABEL)
        neg_path = os.path.join(self.train_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read().decode('utf8')
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()
       
        #for w in self.class_word_counts[POS_LABEL]:
            #print w, self.class_word_counts[POS_LABEL][w]

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))
        

    def update_model(self, bow, label):
        
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        self.class_total_doc_counts[label] +=1
        for word in bow:
            self.class_word_counts[label][word] +=bow[word]
            self.class_total_word_counts[label] +=bow[word]
            self.vocab.add(word)
        
       

    def tokenize_and_update_model(self, doc, label):
        """
        Implement me!

        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """
        bow_dic = self.tokenize_doc(doc)
        self.update_model(bow_dic,label)

    def top_n(self, label, n):
        label_word_counts = self.class_word_counts[label]
        mydict = ast.literal_eval(json.dumps(label_word_counts))
        sorted_list = sorted(mydict.items(),key = operator.itemgetter(1),reverse = True)
        return sorted_list[0:n]
        """
        Implement me!
        Returns the most frequent n tokens for documents with class 'label'.
        """
        
    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label
        according to this NB model.
        """
        # no_of_w_in_y  /  no_all_words_in_y
        res = self.class_word_counts[label][word]/self.class_total_word_counts[label]
        return res
    
        

    def p_word_given_label_and_alpha(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - smoothing parameter
        """
        #print(self.class_word_counts[label][word])
        #print(alpha)
        nom = self.class_word_counts[label][word] + alpha
        denom = self.class_total_word_counts[label] + (len(self.vocab) * alpha)
        
        return nom/denom
        

    def log_likelihood(self, bow, label, alpha):
        """
        Implement me!

        Computes the log likelihood of a set of words given a label and smoothing.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; smoothing parameter
        """
        result = 0
        for word in bow:
            result +=math.log(self.p_word_given_label_and_alpha(word, label, alpha))
        return result

    def log_prior(self, label):
        """
        Implement me!

        Returns the log prior of a document having the class 'label'.
        """
        prior = self.class_total_word_counts[label]/(self.class_total_word_counts[POS_LABEL]+self.class_total_word_counts[NEG_LABEL])
        return math.log(prior)

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!

        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return self.log_prior(label) + self.log_likelihood(bow,label, alpha)
            
    def classify(self, bow, alpha):
        """
        Implement me!

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """
        pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)  
        if pos >= neg:
            return POS_LABEL
        else:
            return NEG_LABEL
        

    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_alpha( word,POS_LABEL,alpha)/self.p_word_given_label_and_alpha(word,NEG_LABEL,alpha)

    def evaluate_classifier_accuracy(self, alpha):
        """
        DO NOT MODIFY THIS FUNCTION

        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0
        #print (self.class_total_word_counts[POS_LABEL], self.class_total_word_counts[NEG_LABEL])
        pos =0
        pos_path = os.path.join(self.test_dir, POS_LABEL)
        neg_path = os.path.join(self.test_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read().decode('utf8')
                    bow = self.tokenize_doc(content)
                    if self.classify(bow, alpha) == POS_LABEL:
                        pos +=1
                    if self.classify(bow, alpha) == label:
                        correct += 1.0
                    total += 1.0
        #print 100 *pos/total
        return 100 * correct / total
    
        