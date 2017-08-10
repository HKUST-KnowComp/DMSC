import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy as np
import theano
import theano.tensor as T

from nn import get_activation_by_name, create_optimization_updates, softmax, ReLU, tanh, linear
from nn import Layer, EmbeddingLayer, LSTM, Dropout, apply_dropout, Query_Repr_Layer, IterAttentionLayer
from utils import say, load_embedding_iterator

np.set_printoptions(precision=3)

def read_corpus(path):
    with open(path) as fin:
	lines = fin.readlines()

    segs = [line.strip().split('\t\t') for line in lines]

    tmp_x = [ seg[1].split('<ssssss>') for seg in segs]
    tmp_x = map(lambda doc: filter(lambda sent: sent, doc), tmp_x)

    corpus_x = map(lambda doc: map(lambda sent: sent.strip().split(), doc), tmp_x)
    corpus_y = map(lambda seg: map(lambda rating: int(rating) - 1, seg[0].strip().split()), segs)
    
    return corpus_x, corpus_y

def create_one_batch(ids, x, y, scale):
    max_len = 0
    for iid in ids:
	for sent in x[iid]:
	    if max_len < len(sent):
		max_len = len(sent)
    batch_x = map(lambda iid: np.asarray(map(lambda sent: sent + [-1] * (max_len - len(sent)), x[iid]), dtype = np.int32).T, ids)
    batch_w_mask = map(lambda iid: np.asarray(map(lambda sent: len(sent) * [1] + [0] * (max_len - len(sent)), x[iid]), dtype = np.float32).T, ids)
    batch_w_len = map(lambda iid: np.asarray(map(lambda sent: len(sent), x[iid]), dtype = np.float32) + np.float32(1e-4), ids)
    #sentence-level input
    batch_x = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_x)
    batch_w_mask = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_w_mask)
    batch_w_len = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 0), batch_w_len)
    
    #review-level input
    batch_y = np.array( [ y[i][0] for i in ids ] )
    batch_ay = np.array( [ y[i][1:] for i in ids ] )
    batch_aay = np.array( [ 1 if ind == cy else 0 for i in ids for cy in y[i][1:] for ind in range(scale)])
    batch_aay = batch_aay.reshape((len(ids), len(y[0][1:]), scale))
    #batch_aay = np.transpose(batch_aay, (0, 2, 1))
    batch_ay_mask = np.array([ 0 if cy < 0 else 1 for i in ids for cy in y[i][1:]], dtype = np.float32)
    batch_ay_mask = batch_ay_mask.reshape((len(ids), len(y[0][1:])))
    
    return batch_x, batch_y, batch_ay, batch_aay, batch_ay_mask, batch_w_mask, batch_w_len, max_len

def create_batches(perm, x, y, batch_size, scale):
    lst = sorted(perm, key=lambda i: len(x[i]))
    batches_x = [ ]
    batches_w_mask = []
    batches_w_len = []
    batches_sent_maxlen = []
    batches_sent_num = []
    batches_y = []
    batches_ay = []
    batches_aay = []
    batches_ay_mask = []
    size = batch_size
    
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by, bay, baay, baym, bm, bl, ml = create_one_batch(ids, x, y, scale)
            batches_x.append(bx)
            batches_y.append(by)
	    batches_ay.append(bay)
	    batches_aay.append(baay)
	    batches_ay_mask.append(baym)
	    batches_w_mask.append(bm)
	    batches_w_len.append(bl)
	    batches_sent_num.append(len(x[ids[0]]))
	    batches_sent_maxlen.append(ml)
            ids = [ i ]
    bx, by, bay, baay, baym, bm, bl, ml = create_one_batch(ids, x, y, scale)
    batches_x.append(bx)
    batches_y.append(by)
    batches_ay.append(bay)
    batches_aay.append(baay)
    batches_ay_mask.append(baym)
    batches_w_mask.append(bm)
    batches_w_len.append(bl)
    batches_sent_num.append(len(x[ids[0]]))
    batches_sent_maxlen.append(ml)
    
    # shuffle batches
    perm = range(len(batches_x))
    random.shuffle(perm)
    batches_x = [ batches_x[i] for i in perm ]
    batches_y = [ batches_y[i] for i in perm ]
    batches_ay = [ batches_ay[i] for i in perm ]
    batches_aay = [batches_aay[i] for i in perm]
    batches_ay_mask = [batches_ay_mask[i] for i in perm]
    batches_w_mask = [ batches_w_mask[i] for i in perm ]
    batches_w_len = [ batches_w_len[i] for i in perm ]
    batches_sent_maxlen = [ batches_sent_maxlen[i] for i in perm ]
    batches_sent_num = [ batches_sent_num[i] for i in perm ]

    return batches_x, batches_y, batches_ay, batches_aay, batches_ay_mask,  batches_w_mask, batches_w_len, batches_sent_maxlen, batches_sent_num

class Model:
    def __init__(self, args, embedding_layer, num_aspects, query):
        self.args = args
        self.embedding_layer = embedding_layer
        self.num_aspects = num_aspects
	self.query = query

    def ready(self):
        args = self.args
        embedding_layer = self.embedding_layer
	num_aspects = self.num_aspects        

        self.n_emb = embedding_layer.n_d
        
	dropout = self.dropout = theano.shared(
                np.float64(args.dropout_rate).astype(theano.config.floatX)
            )

        self.x = T.imatrix('x')
	self.w_masks = T.fmatrix('mask')
	self.w_lens = T.fvector('sent_len')
	self.s_maxlen = T.iscalar('sent_max_len')
	self.s_num = T.iscalar('sent_num')
	self.y = T.ivector('y')
	self.ay = T.imatrix('ay')
	self.ay_mask = T.fmatrix('ay_mask')	
	self.aay = T.itensor3('aay')

        x = self.x
	query = self.query
        
	w_masks = self.w_masks
	w_lens = self.w_lens
	s_ml = self.s_maxlen
	s_num = self.s_num
	n_emb = self.n_emb
	
	y = self.y
        ay = self.ay
	ay_mask = self.ay_mask
	aay = self.aay

	layers = self.layers = [embedding_layer]
        slices  = embedding_layer.forward(x.ravel())
	self.slices = slices = slices.reshape( (x.shape[0], x.shape[1], n_emb) )
	
	slices_query = embedding_layer.forward(query.flatten(), is_node = False)
	slices_query = slices_query.reshape( (query.shape[0], query.shape[1], n_emb))
	
	layers.append(Query_Repr_Layer(slices_query))
	slices_query_tmp = slices_query = layers[-1].forward()
	
	layer = LSTM(n_in = n_emb, n_out = n_emb)
        layers.append(layer)

	prev_output = slices
        prev_output = apply_dropout(prev_output, dropout, v2=True)
        prev_output = layers[-1].forward_all(prev_output, w_masks)

        layer = Layer(n_in = n_emb, n_out = n_emb, activation = tanh)
        layers.append(layer)
        self.slices_query = slices_query = layers[-1].forward(slices_query)

	maskss = []
	w_lenss = []
	for i in range(num_aspects):
	    maskss.append(w_masks)
	    w_lenss.append(w_lens)

	maskss = T.concatenate(maskss, axis = 1)
        w_lenss = T.concatenate(w_lenss)

	layer = IterAttentionLayer(n_in = n_emb, n_out = n_emb)
        layers.append(layer)
	prev_output = layers[-1].forward(prev_output, slices_query, is_word = True, hop = args.hop_word, masks = w_masks, aspect_num = num_aspects)
	prev_output = prev_output.reshape((prev_output.shape[0] * prev_output.shape[1], prev_output.shape[2]))
        prev_output = apply_dropout(prev_output, dropout, v2=True)
	
	prev_output = prev_output.reshape((num_aspects, prev_output.shape[0] / (num_aspects * s_num), s_num, prev_output.shape[1]))
	prev_output = prev_output.dimshuffle(2, 0, 1, 3)
	prev_output = prev_output.reshape((prev_output.shape[0], prev_output.shape[1] * prev_output.shape[2], prev_output.shape[3]))

        layer = LSTM(n_in = n_emb * args.hop_word, n_out = n_emb)
	layers.append(layer)
	prev_output = layers[-1].forward_all(prev_output)
	
	#layers.append(Query_Repr_Layer(slices_query))
        #slices_query = layers[-1].forward()
	layer = Layer(n_in = n_emb, n_out = n_emb, activation = tanh)
        layers.append(layer)
        slices_query = layers[-1].forward(slices_query_tmp) # bug
	
	layer = IterAttentionLayer(n_in = n_emb, n_out = n_emb)
        layers.append(layer)
	prev_output = layers[-1].forward(prev_output, slices_query, is_word = False, hop = args.hop_sent, aspect_num = num_aspects)
        prev_output = prev_output.reshape((prev_output.shape[0] * prev_output.shape[1], prev_output.shape[2]))
	prev_output = apply_dropout(prev_output, dropout, v2=True)

	prev_output = prev_output.reshape((num_aspects, prev_output.shape[0] / num_aspects, prev_output.shape[1]))
	
	softmax_inputs = []
	for i in range(num_aspects):
	    softmax_inputs.append(prev_output[i])
	
	size = n_emb * args.hop_sent
	
	p_y_given_a = []
	pred_ay = []
	nll_loss_ay = []
	
	for i in range(num_aspects):
	    layers.append(Layer(n_in = size,
                    n_out = args.score_scale,
                    activation = softmax,
                    has_bias = False,))

	    p_y_given_a.append(layers[-1].forward(softmax_inputs[i]))
	    nll_loss_ay.append( T.mean(T.sum( -T.log(p_y_given_a[-1]) * aay[:, i, :] * ay_mask[:, i].dimshuffle(0, 'x'))))
	    pred_ay.append(T.argmax(p_y_given_a[-1], axis = 1))

	self.p_y_given_a = p_y_given_a
	self.nll_loss_ay = T.sum(nll_loss_ay)
	self.pred_ay = T.stack(pred_ay).dimshuffle(1, 0)
        
	for l,i in zip(layers[4:], range(len(layers[3:]))):
            say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
           ))
	
	self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))


    def save_model(self, path, args):
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args = pickle.load(fin)
	
        self.ready()
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    def eval_accuracy(self, preds_ay, golds_ay):
	error_ay = sum([(p - y)**2  if y >= 0 else 0 for pss, yss in zip(preds_ay, golds_ay) for ps, ys in zip(pss, yss) for p, y in zip(ps, ys) ]) + 0.0
	fine_ay = sum([ 1  if y == p else 0 for pss, yss in zip(preds_ay, golds_ay) for ps, ys in zip(pss, yss) for p, y in zip(ps, ys) ]) + 0.0
	tot_ay = sum([ 1.0 if y >= 0 else 0.0 for b_ay in golds_ay for ay in b_ay for y in ay])
	return math.sqrt(error_ay / tot_ay), fine_ay / tot_ay

    def train(self, train, dev, test):
        args = self.args
        x_train, y_train = train
        batch = args.batch
	test_batch = args.test_batch
	score_scale = args.score_scale        

	if dev:
	    x_dev_batches, y_dev_batches, ay_dev_batches, ayy_dev_batches, ay_mask_dev_batches, w_mask_dev_batches, w_len_dev_batches, sent_maxlen_dev_batches, sent_num_dev_batches= create_batches(
		    range(len(dev[0])),
		    dev[0],
		    dev[1],
		    test_batch,
		    score_scale
		)

        if test:
	    x_test_batches, y_test_batches, ay_test_batches, ayy_test_batches, ay_mask_test_batches, w_mask_test_batches, w_len_test_batches, sent_maxlen_test_batches, sent_num_test_batches = create_batches(
		    range(len(test[0])),
		    test[0],
		    test[1],
		    test_batch,
		    score_scale
		)

	cost = self.l2_sqr + self.nll_loss_ay

	print 'Building graph...'
	updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_model = theano.function(
             	inputs = [self.x, self.y, self.ay, self.aay, self.ay_mask, self.w_masks, self.w_lens, self.s_maxlen, self.s_num],
             	outputs = [ cost, gnorm ],
             	updates = updates,
             	allow_input_downcast = True
        	)

        eval_acc = theano.function(
             	inputs = [self.x, self.w_masks, self.w_lens, self.s_maxlen, self.s_num],
             	outputs = [self.pred_ay], #, self.output],
             	allow_input_downcast = True
        	)
	
        unchanged = 0
        best_dev_result = 0.0
        dropout_rate = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(x_train))

        say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")
        
	if args.load:
	    self.dropout.set_value(0.0)
	    preds = [ eval_acc( x, wm, wl, sm, sn ) for x, wm, wl, sm, sn in zip(x_dev_batches, w_mask_dev_batches, w_len_dev_batches, sent_maxlen_dev_batches, sent_num_dev_batches)]
            
	    ay_pred = [ pred[0] for pred in preds ]
	    results = self.eval_accuracy(ay_pred, ay_dev_batches)
            
	    best_dev_result = results[1]
            say("\tDEV RMSE/BEST_ACCUARCY/ACCURACY=%.4f_%.4f_%.4f\n" % (
                results[0],
                best_dev_result,
                results[1]
            ))

            preds = [ eval_acc( x, wm, wl, sm, sn ) for x, wm, wl, sm, sn in zip(x_test_batches, w_mask_test_batches, w_len_test_batches, sent_maxlen_test_batches, sent_num_test_batches)]
            ay_pred = [ pred[0] for pred in preds ]
            results = self.eval_accuracy(ay_pred, ay_test_batches)
            say("\tTEST RMSE/ACCURACY=%.4f_%.4f\n" % (
                   results[0],
                   results[1],
             ))

	for epoch in xrange(args.max_epochs):
            self.dropout.set_value(dropout_rate)
	    unchanged += 1
            if unchanged > 20: return
            train_loss = 0.0

            random.shuffle(perm)
	    x_batches, y_batches, ay_batches, aay_batches, ay_mask_batches, w_mask_batches, w_len_batches, sent_maxlen_batches, sent_num_batches = create_batches(perm, x_train, y_train, batch, score_scale)
            
	    N = len(x_batches)
            for i in xrange(N):

                if (i + 1) % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = x_batches[i]
                y = y_batches[i]
		
		va, grad_norm = train_model(x, y, ay_batches[i], aay_batches[i], ay_mask_batches[i], w_mask_batches[i], w_len_batches[i], sent_maxlen_batches[i], sent_num_batches[i])
                train_loss += va

                # debug
                if math.isnan(va):
                    return

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    self.dropout.set_value(0.0)

                    say( "\n" )
                    say( "Epoch %.3f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            float(grad_norm),
                            (time.time()-start_time) / 60.0
                    ))
                    say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")

                    if dev:
            		preds = [ eval_acc(x, wm, wl, sm, sn) for x, wm, wl, sm, sn in zip(x_dev_batches, w_mask_dev_batches, w_len_dev_batches, sent_maxlen_dev_batches, sent_num_dev_batches)]
			ay_pred = [ pred[0] for pred in preds ]
			results = self.eval_accuracy(ay_pred, ay_dev_batches)
                        say("\tDEV RMSE/BEST_ACCUARCY/ACCURACY=%.4f_%.4f_%.4f\n" % (
                            results[0],
                            best_dev_result,
                            results[1]
                        ))

                        if results[1] > best_dev_result:
                            unchanged = 0
                            best_dev_result = results[1]
                            if args.save:
                                self.save_model(args.save, args)

			    preds = [ eval_acc(x, wm, wl, sm, sn) for x, wm, wl, sm, sn in zip(x_test_batches, w_mask_test_batches, w_len_test_batches, sent_maxlen_test_batches, sent_num_test_batches)]
			    ay_pred = [ pred[0] for pred in preds ]
			   
			    results_test = self.eval_accuracy(ay_pred, ay_test_batches)
                            say("\tTEST RMSE/ACCURACY=%.4f_%.4f\n" % (
                                results_test[0],
                                results_test[1]
                                ))

                        if best_dev_result > results[0] + 0.2:
                            return

                    self.dropout.set_value(dropout_rate)
                    start_time = time.time()

def load_doc_corpus(embedding_layer, file_in):
    corpus_x, corpus_y = read_corpus(file_in)
    corpus_x = map(lambda doc: map(lambda sent: embedding_layer.map_to_ids(sent).tolist(), doc), corpus_x)
    return corpus_x, corpus_y

def load_lis(file_in):
    result = []
    with open(file_in, 'r') as fin:
	for line in fin.readlines():
	    if line.strip():
		result.append(line.strip())
    return result

def main(args):
    assert args.train, "Training  set required"
    assert args.dev, "Dev set required"
    assert args.test, "Test set required"
    assert args.emb, "Pre-trained word embeddings required."
    assert args.aspect_seeds, "Aspect seeds required."
	
    print args

    seeds = load_lis(args.aspect_seeds)
    say("loaded {} aspect seeds\n".format(len(seeds)))

    embedding_layer = EmbeddingLayer(
                n_d = 100,
                vocab = [ "<unk>" ],
                pre_embs = load_embedding_iterator(args.emb),
            )

    seeds_id = np.array(map(lambda seed: embedding_layer.map_to_ids(seed.strip().split()).tolist(), seeds), dtype = np.int32)

    if args.train:
	train_x, train_y = load_doc_corpus(embedding_layer, args.train)

    if args.dev:
	dev_x, dev_y = load_doc_corpus(embedding_layer, args.dev)

    if args.test:
	test_x, test_y = load_doc_corpus(embedding_layer, args.test)
    
    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    num_aspects = len(seeds_id),
		    query = seeds_id
            )
	if args.load:
	    print 'loading model...'
	    model.load_model(args.load)
        else:
	    model.ready()
	
	print 'training...'
        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y) if args.test else None
            )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aspect_seeds",
            type = str,
            default = "",
            help = "path to aspect seeds"
        )
    argparser.add_argument("--batch",
            type = int,
            default = 3,
            help = "mini-batch size"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to dev data"
	)
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0.3,
            help = "dropout rate"
        )
    argparser.add_argument("--emb",
            type = str,
            default = "",
	    help = "path to pre_trained embedding"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = 500,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--hop_sent",
            type = int,
            default = 2,
	    help = "hop for sentence level"
        )  
    argparser.add_argument("--hop_word",
            type = int,
            default = 2,
            help = "hop for word level"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adadelta",
            help = "learning method (sgd, adagrad, adam, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.001",
            help = "learning rate"
        )
    argparser.add_argument("--load",
            type = str,
            default = "",
            help = "load model from this file"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--test_batch",
            type = int,
            default = 3,
            help = "test mini-batch_size"
        )
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--save",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--score_scale",
	    type = int,
	    default = 5,
	    help = "score scale"
	)
    args = argparser.parse_args()
    main(args)


