import tensorflow as tf
import numpy as np

class CWCNN(object):
    """
    A CWCNN for text classification
    1.the first layer of cnn to deal with char embedding
    input: the char embedding matrix of every word in a sentence
    output: an embedding of character level features
    2.the second layer of cnn to deal with the concatenated embedding
    input: the first layer's output concatenate(+) word embedding
    output: cnn output
    3.max-pooling layer
    4.softmax layer
    """
    def __init__(self, sess,
                 W, C,
                 word_idx_map,
                 char_idx_map,
                 word_embed_dim = 300,
                 char_embed_dim = 15,
                 max_sent_length = 65,
                 max_word_length = 40,
                 w_filter_hs = [3,4,5],
                 c_filter_hs = [1,2,3,4,5],
                 hidden_units = [10,100,2],
                 batch_size = 50,
                 epoch = 25,
                 dropout_rate = 0.5,
                 lr_decay = 0.95,
                 sqr_norm_lim = 9):
        """
        Initialize the parameters for CWCNN model
        Args:
        U: the initialized word vector matrix
        batch_size: size of batch per epoch
        epoch: number of epoch
        word_embed_dim: the dimension of word embeddings
        char_embed_dim: the dimension of char embeddings
        filter_hs: the window sizes of filters
        hidden_units: feature maps of every kind of filter window size[c-layer,w-layer,softmax-layer]
        dropout_rate: the probability of dropout
        lr_decay: adadelta decay parameter or learning rate decay???
        sqr_norm_lim: limit the l2-norm of gradient to 3
        """
        self.sess = sess
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        
        #c-layer
        self.char_idx_map = char_idx_map
        self.char_embed_dim = char_embed_dim
        self.max_word_length = max_word_length
        self.c_filter_hs = c_filter_hs
        self.c_hidden_unit = hidden_units[0]
        
        #w-layer
        self.word_idx_map = word_idx_map
        self.word_embed_dim = word_embed_dim
        self.max_sent_length = max_sent_length
        self.w_filter_hs = w_filter_hs
        self.w_hidden_unit = hidden_units[1]
        
        #softmax
        self.s_hidden_unit = hidden_units[2]
        self.dropout_rate = dropout_rate
        self.sqr_norm_lim = sqr_norm_lim
        
        with tf.variable_scope("CWCNN"):
            #element:every word's c_lever_embedding in a sentence of that batch
            self.char_inputs = []
            #every word's conbined(char and word) embedding of a batch
            self.word_inputs = []
            
            self.cnn_outputs = []
            
            #char_W = tf.get_variable("char_W",[self.char_vocab_size, self.char_embed_dim],initialize = C)
            with tf.variable_scope("c-layer") as scope:
                #char_input is a tensor of index
                self.char_input = tf.placeholder(tf.int32, [None, self.max_sent_length, self.max_word_length],name = "char_input")
                #char_indices is a list of tf.tensor
                #[<tf.Tensor 'split:0' shape=(None<batch_size>,1, max_word_length) dtype = int32>, <...> ...]
                char_indices = tf.split(1,self.max_sent_length, self.char_input)
                #for idx in xrange(len(char_indices)): is the same
                for idx in xrange(self.max_sent_length):
                    #char_index shape:[batch_size, max_word_length]
                    char_index = tf.reshape(char_indices[idx], [-1, self.max_word_length])
                    #??????????????????????????????????
                    if idx != 0:
                        scope.reuse_variables()
                    #shape [None, max_word_length, char_embed_dim]
                    char_embeds = tf.nn.embedding_lookup(C,char_index)
                    #shape [None, max_word_length, char_embed_dim, 1]
                    char_embeds_expanded = tf.expand_dims(char_embeds, -1)
                    
                    #create a convolution + max-pooling layer for each c_filter size
                    c_outputs = []
                    for i, c_filter_h in enumerate(c_filter_hs):
                        with tf.name_scope("conv-maxpool-%s" % c_filter_h):
                            #convolutional layer
                            filter_shape = [c_filter_h, char_embed_dim, 1, self.c_hidden_unit]
                            W = tf.get_variable("W", shape=filter_shape,
                                                initializer = tf.truncated_normal_initializer(stddev = 0.1))
                            b = tf.get_variable("b", shape=[self.c_hidden_unit],
                                                initializer = tf.constant_initializer(0.1))
                            conv = tf.nn.conv2d(char_embeds_expanded,
                                                W,
                                                strides = [1,1,1,1],
                                                padding="VALID",
                                                name = "conv")
                            #Apply non-linearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            #Maxpooling over the output
                            #shape:[None(batch_size), 1, 1, c_hidden_unit]
                            pooled = tf.nn.max_pool(h,
                                                    ksize=[1, self.max_word_length - c_filter_h + 1, 1, 1],
                                                    strides = [1,1,1,1],
                                                    padding = "VALID",
                                                    name = "pool")
                            #shape:[None(batch_size), c_hidden_unit]
                            c_output = tf.squeeze(pooled)
                            c_outputs.append(c_output)
                    c_embedding = tf.concat(1,c_outputs)
                    self.char_inputs.append(c_embedding)
            
            #concatenate the c_embedding with word_embedding
            self.word_input = tf.placeholder(tf.int32, [None, self.max_sent_length], name="word_input")
            word_indices = tf.split(1, self.max_sent_length, self.word_input)
            for idx in xrange(self.max_sent_length):
                word_index = word_indices[idx]
                #shape:[None(batch_size), word_embed_dim]
                word_embed = tf.nn.embedding_lookup(W, word_index)
                
                wc = tf.concat(1,[word_embed,self.char_inpus[idx]])
                self.word_inputs.append(wc)
            
            with tf.name_scope("w-layer") as scope:
                b_size, s_embed_dim = self.word_inputs[0].get_shape()[0], self.word_inputs[0].get_shape()[1]
                #word_inputs is a list of 2-d tensor,
                #how to reshape  word_inputs to a 3-d tensor??? use tf.pack() and tf.transpose() function
                #word_inputs_trans is a tensor of shape(b_size, max_sent_length, s_embed_dim)
                self.word_inputs_trans = tf.transport(tf.pack(self.word_inputs), perm=[1,0,2])
                
                #
                
            
                    
                           
                    
                    
                    
                    
                
                
            