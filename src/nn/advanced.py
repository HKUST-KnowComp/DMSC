import numpy as np
import theano
import theano.tensor as T

from .initialization import random_init, create_shared
from .initialization import tanh
from .basic import Layer

class Query_Repr_Layer(Layer):
    def __init__(self, repr_init):
	self.repr_init = theano.shared(repr_init, name = "ReprInit")
	self.create_parameters()
    
    def create_parameters(self):
	repr_init = self.repr_init
	self.lst_params = [self.repr_init]

    def forward(self):
	return self.repr_init

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())

class IterAttentionLayer(Layer):
    def __init__(self, n_in, n_out):
	self.n_in = n_in
	self.n_out = n_out
	self.create_parameters()

    def create_parameters(self):
	n_in = self.n_in
	n_out = self.n_out

	self.W_Pick = create_shared(random_init((n_in, n_out)), name = "W_Pick")
	self.W_Query = create_shared(random_init((n_out)), name = "W_Query")
	self.W_Doc = create_shared(random_init((n_out)), name = "W_Doc")
	self.lst_params = [self.W_Pick, self.W_Query, self.W_Doc]

    def forward(self, prev_output, slices_query, is_word = True, hop = 1, masks = None, aspect_num = 7):
        W_Pick = self.W_Pick
        W_Query = self.W_Query
        W_Doc = self.W_Doc
      
	n_out = self.n_out
        
	doc_vecs = prev_output
        query_vecs = slices_query

	final_repr = []
        
	if is_word:
            doc_vecs = doc_vecs.dimshuffle(1, 0, 2)
        else:
            doc_vecs = doc_vecs.dimshuffle(1, 0, 2)
            doc_vecs = doc_vecs.reshape((query_vecs.shape[0], doc_vecs.shape[0] / query_vecs.shape[0], doc_vecs.shape[1], doc_vecs.shape[2]))

        pick_vec = T.zeros((aspect_num, n_out))
        for i in range(hop):
            pick_vec_tmp = T.dot(pick_vec, W_Pick)
            if i == 0:
                alpha = query_vecs * pick_vec_tmp.dimshuffle(0, 'x', 1)
                alpha = T.exp(T.dot(T.tanh(alpha), W_Query))
                alpha_sum = T.sum(alpha, axis = 1)
                alpha = alpha / alpha_sum.dimshuffle(0, 'x')
                pick_vec_tmp = T.sum(alpha.dimshuffle(0, 1, 'x') * query_vecs, axis = 1)
            else:
                alpha = query_vecs.dimshuffle(0, 1, 'x', 2) * pick_vec_tmp.dimshuffle(0, 'x', 1, 2)
                alpha = T.exp(T.dot(T.tanh(alpha) , W_Query))
                alpha_sum = T.sum(alpha, axis = 1)
                alpha = alpha / alpha_sum.dimshuffle(0, 'x', 1)
                pick_vec_tmp =  T.sum(alpha.dimshuffle(0, 1, 2, 'x') * query_vecs.dimshuffle(0, 1, 'x', 2), axis = 1)

            pick_vec_tmp = T.dot(pick_vec_tmp, W_Pick)

            if i == 0:
                alpha = doc_vecs * pick_vec_tmp.dimshuffle(0, 'x', 'x', 1)
            else:
                alpha = doc_vecs * pick_vec_tmp.dimshuffle(0, 1, 'x', 2)

            alpha = T.exp(T.dot(T.tanh(alpha), W_Doc))
            if masks is not None:
                if masks.dtype != theano.config.floatX:
                    masks = T.cast(masks, theano.config.floatX)
                alpha = alpha * masks.dimshuffle(1, 0)
            alpha_sum = T.sum(alpha, axis = 2)
            alpha = alpha / alpha_sum.dimshuffle(0, 1, 'x')

            pick_vec =  T.sum(alpha.dimshuffle(0, 1, 2, 'x') * doc_vecs, axis = 2)
	    final_repr.append(pick_vec)
        return T.concatenate(final_repr, axis = 2)

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


