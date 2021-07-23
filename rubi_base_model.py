import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from torch.nn import functional as F

import numpy as np
from vqa_debias_loss_functions import *


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4

def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm

class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 dimensions,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears) - 1):
                x = nn.functional.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = nn.functional.dropout(x, self.dropout, training=self.training)
        return x

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier,c_1,c_2,mask_classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = LearnedMixin(0.36)
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.c_1=c_1
        self.c_2=c_2
        self.mask_classifier = mask_classifier


    def forward(self, v, q, labels, bias,v_mask, loss_type = None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)

        q_pred=self.c_1(q_emb.detach())

        q_out=self.c_2(q_pred)


        if labels is not None:
            # rubi_logits=logits*torch.sigmoid(q_pred)
            # loss=F.binary_cross_entropy_with_logits(rubi_logits, labels)+F.binary_cross_entropy_with_logits(q_out, labels)
            # 

            if loss_type == 'lmh_rubi':
                rubi_logits=logits*torch.sigmoid(q_pred)
                loss=self.debias_loss_fn(joint_repr, rubi_logits, bias, labels).mean(0)+F.binary_cross_entropy_with_logits(q_out, labels) * labels.size(1)
            else:
                rubi_logits=logits*torch.sigmoid(q_pred)
                loss=F.binary_cross_entropy_with_logits(rubi_logits, labels)+F.binary_cross_entropy_with_logits(q_out, labels)
                loss *= labels.size(1)

        else:
            loss = None
        return logits, loss, att

def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    c_1=MLP(input_dim=1024,dimensions=[1024,1024,dataset.num_ans_candidates])
    c_2=nn.Linear(dataset.num_ans_candidates,dataset.num_ans_candidates)
    mask_classifier = MLP(input_dim=36, dimensions=[1024, 1024, dataset.num_ans_candidates])

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier,c_1,c_2, mask_classifier)