import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable, Function
import torch.cuda
import torch.nn.functional as F
from pprint import pprint, pformat
import math
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import sys
import s3fs

#from pytorch_visualize import *
#from pytorch_model import ProdLDA

class ProdLDA(nn.Module):

    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        # encoder
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)             # 1995 -> 100
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(ac.num_topic, ac.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)                      # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var    = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        # initialize decoder weight
        if ac.init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, ac.init_mult)
        # remove BN's scale parameters
        self.logvar_bn .register_parameter('weight', None)
        self.mean_bn   .register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)

    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)))             # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss





parser = argparse.ArgumentParser()
parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-t', '--num-topic',        type=int,   default=5)
parser.add_argument('-b', '--batch-size',       type=int,   default=200)
parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
parser.add_argument('-e', '--num-epoch',        type=int,   default=80)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
parser.add_argument('--start',                  action='store_true')        # start training at invocation
parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration

parser.add_argument('-ip', '--input',          type = str)# default = '/Users/ttan/Desktop/17_Fall/BOF/avitm_Tian/smallFinalDF.csv')        # do not use GPU acceleration

args = parser.parse_args()


# default to use GPU, but have to check if GPU exists
if not args.nogpu:
    if torch.cuda.device_count() == 0:
        args.nogpu = True

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data(dataset_tr):
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size
    #dataset_tr = 'data/20news_clean/train.txt.npy'
    #data_tr = np.load(dataset_tr,encoding= 'latin1')
    #dataset_te = 'data/20news_clean/test.txt.npy'
    #data_te = np.load(dataset_te,encoding= 'latin1')
    #vocab = 'data/20news_clean/vocab.pkl'
    #vocab = pickle.load(open(vocab,'rb'),encoding = 'latin1')

    if dataset_tr.startswith('s3://'):
        s3 = s3fs.S3FileSystem(profile_name = 'datascience')
        dataset_tr = dataset_tr.replace('s3://', '')
        print("******************")
        print(dataset_tr)
        with s3.open(dataset_tr, 'rb') as f:
            df = pd.read_csv(dataset_tr)#,encoding='latin-1')
    else:
        df = pd.read_csv(dataset_tr)#,encoding='latin-1')
    cids = list(set(df.cid.values))
    urls = list(set(df.url.values))
    vocab = dict(zip(urls, list(range(len(urls)))))
    distinctCidDF_temp = df.groupby("cid")["url"].apply(lambda x: "|||".join(x))
    distinctCidDF = distinctCidDF_temp.apply(lambda row: np.array([float(vocab[url]) for url in row.split("|||")]))
    data_tr = distinctCidDF.values
    data_te = distinctCidDF.values
    vocab_size=len(vocab)
    #--------------convert to one-hot representation------------------
    print('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print('Data Loaded')
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    if not args.nogpu:
        tensor_tr = tensor_tr.cuda()
        tensor_te = tensor_te.cuda()

def make_model():
    global model
    net_arch = args # en1_units, en2_units, num_topic, num_input
    net_arch.num_input = data_tr.shape[1]
    model = ProdLDA(net_arch)
    if not args.nogpu:
        model = model.cuda()

def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def train():
    for epoch in range(args.num_epoch):
        all_indices = torch.randperm(tensor_tr.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            if not args.nogpu: batch_indices = batch_indices.cuda()
            input = Variable(tensor_tr[batch_indices])
            recon, loss = model(input, compute_loss=True)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))

associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh', 
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}
def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        topics = identify_topic_in_line(line)
        print('|'.join(topics))
        print('     {}'.format(line))
    print('---------------End of Topics------------------')

def print_perp(model):
    cost=[]
    model.eval()                        # switch to testing mode
    input = Variable(tensor_te)
    recon, loss = model(input, compute_loss=True, avg_loss=False)
    loss = loss.data
    counts = tensor_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', math.exp(avg))

def visualize():
    global recon
    input = Variable(tensor_te[:10])
    register_vis_hooks(model)
    recon = model(input, compute_loss=False)
    remove_vis_hooks()
    save_visualization('pytorch_model', 'png')

if __name__=='__main__' and args.start:
    make_data(args.input)
    make_model()
    make_optimizer()
    train()
    emb = model.decoder.weight.data.cpu().numpy().T
    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
    print_perp(model)
    #visualize()

