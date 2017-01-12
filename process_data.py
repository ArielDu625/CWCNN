import sys, re, os
import numpy as np
import cPickle
from collections import defaultdict
import logging

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #return string.strip().lower()
    return string.strip()

def build_data_cv(data_folder, cv = 10, clean_string = True):
    """
    load data and split into 10 folds
    """
    reviews = []
    #vocab is a dictionary, key is the word appeard, value is it's document-frequency
    vocab = defaultdict(float)
    #max sentence length
    max_length = 0
    #chars is a dictionary, key is the char appeard, value is its frequency??
    chars = defaultdict(float)
    #max word length
    max_word_length = 0
    
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    with open(pos_file, 'rb') as f:
        for line in f:
            rev = []
            rev_ch = []
            rev.append(line.strip())            
            if clean_string:
                #" ".join(rev) use " " to concatenate the element of rev
                #for example, rev = ['hello,word']
                #" ".join(rev) return a string 'hello,word'
                original_rev = clean_str(" ".join(rev))
            else:
                original_rev = " ".join(rev)
            rev_ch.append(original_rev)
            for r in rev_ch:
                for ch in r:
                    chars[ch] += 1
            
            max_length = max(max_length, len(original_rev))
            words = set(original_rev.split())
            for word in words:
                max_word_length = max(max_word_length, len(word))
                vocab[word] += 1
            datum = {"y": [0,1],
                     "text": original_rev,
                     "num_words": len(original_rev.split()),
                     "split": np.random.randint(0,cv)}
            reviews.append(datum)
    with open(neg_file) as f:
        for line in f:
            rev = []
            rev_ch = []
            rev.append(line.strip())
            if clean_string:
                original_rev = clean_str(" ".join(rev))
            else:
                original_rev = " ".join(rev)
            rev_ch.append(original_rev)
            for r in rev_ch:
                for ch in r:
                    chars[ch] += 1
            
            max_length = max(max_length, len(original_rev))
            words = set(original_rev.split())
            for word in words:
                max_word_length = max(max_word_length, len(word))
                vocab[word] += 1
            datum = {"y": [1,0],
                     "text": original_rev,
                     "num_words": len(original_rev),
                     "split": np.random.randint(0,cv)}
            reviews.append(datum)
    return reviews, vocab, max_length, chars, max_word_length

def load_word_vector(w2v_file, vocab):
    """
    Load 300x1 word vectors from Google word2vec
    """
    word_vectors = {}
    with open(w2v_file, "rb") as f:
        #the first line of the file is vocab_size and layer1_size?
        #what does the layer1_size mean?
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vectors[word] = np.fromstring(f.read(binary_len), dtype = 'float32')
            else:
                # the word not appeard in the dataset, skip this word vector
                f.read(binary_len)
    return word_vectors
 
def add_unknown_word_vector(word_vectors, vocab, min_df = 1, k = 300):
    """
    For words that occur in least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vectors and vocab[word] >= min_df:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, k)

def get_W(word_vectors, k = 300):
    """
    Get word matrix, W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vectors)
    word_idx_map = dict()
    W = np.zeros([vocab_size + 1, k], dtype = 'float32')
    W[0] = np.zeros(k, dtype = 'float32')
    i = 1
    for word in word_vectors:
        W[i] = word_vectors[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def initialize_char_vector(chars, dimension = 15):
    char_vectors = {}
    for char in chars:
        char_vectors[char] = np.random.uniform(-0.25, 0.25, dimension)
    return char_vectors

def get_C(char_vectors, k = 15):
    """
    Get char matrix, C[i] is the vector for c indexed by i
    """
    chars_size = len(char_vectors)
    char_idx_map = dict()
    C = np.zeros([chars_size + 1, k], dtype = 'float32')
    C[0] = np.zeros(k, dtype = 'float32')
    i = 1
    for char in char_vectors:
        C[i] = char_vectors[char]
        char_idx_map[char] = i
        i += 1
    return C, char_idx_map
 
 
if __name__ == "__main__":
    w2v_file = "./data/GoogleNews-vectors-negative300.bin"
    data_folder = ["./data/rt-polarity.pos", "./data/rt-polarity.neg"]
    
    log_folder = "./logs"
    log_file =  "./logs/data_process.log"
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
    
    reviews, vocab, max_length, chars, max_word_length = build_data_cv(data_folder, cv = 10, clean_string = True)
    logger.info("Number of reviews: %d" % (len(reviews)))
    logger.info("Vocab size: %d" % (len(vocab)))
    logger.info("Max sentence length: %d" % (max_length))
    logger.info("Chars size: %d" % (len(chars)))
    logger.info("Max word length: %d" % (max_word_length))
    
    # word_vectors is a dictionary, key is the word, value is the word vector of the word
    word_vectors = load_word_vector(w2v_file, vocab)
    logger.info("Number of words already in w2v: %d" % (len(word_vectors)))
    add_unknown_word_vector(word_vectors, vocab)
    # word_idx_map is a dictionary, key is the word, value is the index in matrix W
    # W is a word vector matrix
    W, word_idx_map = get_W(word_vectors)
    
    #char_vectors is a dictionary ,key is the char, value is the char vector of the char, default dimension is 15
    char_vectors = initialize_char_vector(chars)
    C, char_idx_map = get_C(char_vectors)
    
    
    cPickle.dump([reviews, W, word_idx_map, vocab, max_length, C, char_idx_map, chars, max_word_length], open("./data/mr.p", "wb"))
    logger.info("dataset process done!")
    logger.info("[reviews, W, word_idx_map, vocab, max_length, C, char_idx_map, chars, max_word_length] has been saved to file mr.p")