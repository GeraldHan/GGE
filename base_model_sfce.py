import torch
import torch.nn as nn
from attention import Attention, NewAttention, SelfAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
import numpy as np
from torch.nn import functional as F
from vqa_debias_loss_functions import LearnedMixin


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_att, q_net, q_net_2, v_net, classifier, c_1,c_2):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_att = q_att
        self.q_net = q_net
        self.q_net_2 = q_net_2
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.c_1=c_1
        self.c_2=c_2
        self.vision_lin = torch.nn.Linear(1024, 1)
        self.question_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, q, labels, bias, v_mask, q_mask, loss_type = None):
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

        joint_repr = v_repr * q_repr
        logits = self.classifier(joint_repr)

        q_pred=self.c_1(q_emb.detach())

        q_out=self.c_2(q_pred)

        q_out = F.softmax(q_out, dim=1)
        logits = F.softmax(logits, dim=1)


        if labels is not None:
            if loss_type == 'q':
                loss = F.binary_cross_entropy(q_out, labels)

            elif loss_type == 'joint':
                ref_logits = q_out 
                loss = self.debias_loss_fn(None, logits, ref_logits, labels)

            elif loss_type == 'q_bias':
                loss_q = F.binary_cross_entropy(q_out, labels)
                ref_logits = q_out
                loss = self.debias_loss_fn(None, logits, ref_logits, labels) + loss_q

            else:
                loss = self.debias_loss_fn(joint_repr, logits, bias, labels).mean(0)
        else:
            loss = None

        return logits, loss, att

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
    q_att = SelfAttention(q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    q_net_2 = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    c_1=MLP(input_dim=q_emb.num_hid,dimensions=[1024,1024,dataset.num_ans_candidates])
    c_2=nn.Linear(dataset.num_ans_candidates,dataset.num_ans_candidates)

    return BaseModel(w_emb, q_emb, v_att, q_att, q_net, q_net_2, v_net, classifier, c_1, c_2)
