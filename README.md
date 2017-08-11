# DMSC_MC
This repository is for our EMNLP17 paper "Document-Level Multi-Aspect Sentiment Classification as Machine Comprehension". We thank Tao Lei as our code is developed based on [his code](https://github.com/taolei87/rcnn/tree/master/code).

## Setup
git clone git@github.com:HKUST-KnowComp/DMSCMC.git

cd src

export PYTHONPATH=/path/to/DMSCMC/src

## Usage
- Run experiments
'''
cd src
THEANO_FLAGS="floatX=float32,device=gpu0,cuda.root=/usr/local/cuda,on_unused_input=ignore,optimizer=fast_compile"  python DMSCMC.py --emb  ../data/tripadvisor/embs  --train  ../data/tripadvisor/train  --dev  ../data/tripadvisor/dev --test  ../data/tripadvisor/test --save ../model/base  --aspect_seeds ../data/tripadvisor/aspect.10.words 
'''
- Run pre-trained model
'''
cd src
THEANO_FLAGS="floatX=float32,device=gpu0,cuda.root=/usr/local/cuda,on_unused_input=ignore,optimizer=fast_compile"  python DMSCMC.py --emb  ../data/tripadvisor/embs  --train  ../data/tripadvisor/train  --dev  ../data/tripadvisor/dev --test  ../data/tripadvisor/test --save ../model/base  --aspect_seeds ../data/tripadvisor/aspect.10.words --model ../model/base
'''

## Environment
* Python 2.7 
* CUDA
* [Theano](http://deeplearning.net/software/theano/) >= 0.7
* [Numpy](http://www.numpy.org) 


