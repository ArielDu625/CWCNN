import re, sys, time
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import tensorflow as tf
import logging

def get_idx_from_sent(sent, word_idx_map, max_l = 56, filter_h = 5):
    """
    Transforms sentence into a list of indices. Pad with zeros
    """
    x = []
    pad = filter_h - 1
    for _ in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        #since the add_unknown_word_vector operation, every word is in word_idx_map
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2*pad:
        x.append(0)
    return x

def make_idx_data_cv(reviews, word_idx_map, char_idx_map, max_l = 56, filter_h = 5):
    """
    Transform sentences into a 2-d matrix
    """
    train, test = [], []
    for review in reviews:
        sent = get_idx_from_sent(review['text'], word_idx_map, max_l, filter_h)
        sent.append(review['y'])
        if review['split'] == cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train, dtype = "int")
    test = np.array(test, dtype = "int")
    return [train, test]
   
if __name__ == '__main__':
    log_folder = "./logs"
    log_file =  "./logs/train.log"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode = 'w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d]:%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("loading data...")
    x = cPickle.load(open("./data/mr.p", "rb"))
    reviews, W, word_idx_map, vocab, max_length, C, char_idx_map, chars, max_word_length = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
    logger.info("data loaded!")

    results = []
    r = range(0,10)
    for i in r:
        datasets = make_idx_data_cv(reviews, word_idx_map, char_idx_map, i,
                                    max_l = 56, filter_h = 5)
        #define a CWCNN model
        model = CWCNN(sess,
                      W, C,
                      word_idx_map,
                      char_idx_map,
                      word_embed_dim = 300,
                      char_embed_dim = 15,
                      max_sent_length = max_length,
                      max_word_length = max_word_length,
                      w_filter_hs = [3,4,5],
                      c_filter_hs = [1,2,3,4,5],
                      hidden_units = [50,100,2],
                      batch_size = 50,
                      epoch = 25,                     
                      dropout_rate = 0.5,
                      lr_decay = 0.95,
                      sqr_norm_lim = 9)
        model.train(datasets[0])
        perf = model.test(datasets[1])
        logger.info("corss-validation[%d]-Test Performance:%.5f" % (i, perf))
        results.append(perf)
    logger.info("final 10 fold cross-validation mean performance: %.5f" %(np.mean(results)))