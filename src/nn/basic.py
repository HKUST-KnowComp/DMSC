import numpy as np
import theano
import theano.tensor as T

from utils import say
from .initialization import default_srng, default_rng, USE_XAVIER_INIT
from .initialization import set_default_rng_seed, random_init, create_shared
from .initialization import ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name

class Dropout(object):
    def __init__(self, dropout_prob, srng=None, v2=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape,
                dtype = theano.config.floatX
            )
        return x * mask / d


def apply_dropout(x, dropout_prob, v2=False):
    return Dropout(dropout_prob, v2=v2).forward(x)

class Layer(object):
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True,
			    scale = 1):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias
	self.scale = scale
        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation)

    def initialize_params(self, n_in, n_out, activation):
        if USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001 * scale).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                W_vals *= (0.001 * self.scale)
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())


class RecurrentLayer(Layer):
    def __init__(self, n_in, n_out, activation,
            clip_gradients=False):
        super(RecurrentLayer, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in super-class Layer
        self.initialize_params(n_in + n_out, n_out, activation)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(
                T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b
            )

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h

class EmbeddingLayer(object):
    def __init__(self, n_d, vocab, oov="<unk>", pre_embs=None, fix_init_embs=False):
        if pre_embs is not None:
            vocab_map = {}
	    words = []
            embs = [ ]
            for word, vector in pre_embs:
                if word in vocab_map:
		    continue
		vocab_map[word] = len(vocab_map)
                embs.append(vector)
                words.append(word)

            self.init_end = len(embs) if fix_init_embs else -1
            
	    if n_d != len(embs[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(embs[0]), len(embs[0])
                    ))
                n_d = len(embs[0])

            say("{} pre-trained embeddings loaded.\n".format(len(embs)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    embs.append(random_init((n_d,))*(0.01 if word != oov else 0))
                    words.append(word)

            self.emb_vals = np.vstack(embs).astype(theano.config.floatX)
            self.vocab_map = vocab_map
            self.words = words
        else:
            words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    words.append(word)

            self.words = words
            self.vocab_map = vocab_map
            self.emb_vals = random_init((len(self.vocab_map), n_d)) * 0.01
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1
	
        self.embs = create_shared(self.emb_vals)
        if self.init_end > -1:
            self.embs_trainable = self.embs[self.init_end:]
        else:
            self.embs_trainable = self.embs

        self.n_vocab = len(self.vocab_map)
        self.n_d = n_d

    def map_to_word(self, id):
	n_vocab, words = self.n_vocab, self.words
	return words[id] if id < n_vocab else '<err>' 

    def map_to_words(self, ids):
        n_vocab, words = self.n_vocab, self.words
        return [ words[i] if i < n_vocab else "<err>" for i in ids ]
    
    def map_to_ids(self, words, filter_oov=False):
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x != oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x, is_node = True):
	if is_node:
            return self.embs[x]
	else:
	    return self.emb_vals[x]

    @property
    def params(self):
        return [ self.embs_trainable ]

    @params.setter
    def params(self, param_list):
        self.embs.set_value(param_list[0].get_value())


class LSTM(Layer):
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False, direction = 'Bi'):

        self.n_in = n_in
        self.n_out = n_out
        n_out_t = self.n_out_t = n_out
	
	if direction == 'Bi':
	    n_out_t = self.n_out_t = n_out / 2
	
	self.activation = activation
        self.clip_gradients = clip_gradients
	self.direction =  direction
        self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)
	
	if direction == "Bi":
	    self.in_gate = RecurrentLayer(n_in, n_out_t , sigmoid, clip_gradients)
            self.forget_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.out_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.input_layer = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)
	    self.in_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.forget_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.out_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.input_layer_b = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)

        self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.forget_gate , self.out_gate ]
	
	if direction == "Bi":
	    self.internal_layers = [self.input_layer_b, self.input_layer, self.in_gate_b, self.in_gate, 
				    self.forget_gate_b, self.forget_gate, self.out_gate_b, self.out_gate]

    def forward(self, x, mask, hc):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate.forward(x,h_tm1)
        forget_t = self.forget_gate.forward(x,h_tm1)
        out_t = self.out_gate.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x,h_tm1)
	c_t = c_t * mask.dimshuffle(0, 'x')
	c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
	h_t = h_t * mask.dimshuffle(0, 'x')
	h_t = T.cast(h_t, 'float32')

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def backward(self, x, mask, hc):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate_b.forward(x,h_tm1)
        forget_t = self.forget_gate_b.forward(x,h_tm1)
        out_t = self.out_gate_b.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer_b.forward(x,h_tm1)
        c_t = c_t * mask.dimshuffle(0, 'x')
	c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
        h_t = h_t * mask.dimshuffle(0, 'x')
	h_t = T.cast(h_t, 'float32')

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def forward_all(self, x, masks = None, h0=None, return_c=False):
	n_out_t = self.n_out_t
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], n_out_t * 2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((n_out_t * 2,), dtype=theano.config.floatX)
        if masks is None:
	    masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)
	
	h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
                    outputs_info = [ h0 ]
                )
	if self.direction == "Bi":
	    if x.ndim > 1:
	    	h1, _ = theano.scan(
			fn = self.backward,
			sequences = [x[::-1, ::, ::], masks[::-1, ::]],
			outputs_info = [h0]	
		    )
	    	h = T.concatenate((h, h1[::-1, ::, ::][:, :, n_out_t:]), axis = 2)
	    else:
                h1, _ = theano.scan(
                        fn = self.backward,
                        sequences = [x[::-1, ::], masks[::-1]],
                        outputs_info = [h0]
                    )
                h = T.concatenate((h, h1[::-1, ::][:, n_out_t:]), axis = 1)

        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,n_out_t:]
        else:
            return h[:,n_out_t:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

