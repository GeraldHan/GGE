import torch
import torch.nn as nn
from attention import Attention, NewAttention, SelfAttention, BiAttention, ApplyAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
import numpy as np
from torch.nn import functional as F
from vqa_debias_loss_functions import LearnedMixin
from torch.nn.utils.weight_norm import weight_norm


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, attention, q_net, v_net, classifier, c_1,c_2, apply_attention):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.attention = attention
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        self.c_1=c_1
        self.c_2=c_2
        self.apply_att = apply_attention

    def forward(self, v, q, labels, bias, v_mask, q_mask, loss_type = None):
        """Forward
 
        v: [batch, num_objs, obj_dim] 
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        atten, logits = self.attention(v, q_hidden, None, q_mask) # batch x glimpses x v_num x q_num

        joint_repr = self.apply_att(v, q_hidden, None, q_mask, atten, logits)

        logits = self.classifier(joint_repr)

        q_pred=self.c_1(q_emb.detach())

        q_out=self.c_2(q_pred)

        if labels is not None:

            if loss_type == 'q':                
                loss = self.debias_loss_fn(None, q_out, bias, labels)

            elif loss_type == 'joint':
                ref_logits = torch.sigmoid(q_pred) + bias
                loss = self.debias_loss_fn(None, logits, ref_logits, labels)
                # y_gradient = 2 * labels * torch.sigmoid(-2 * labels * ref_logits)
                # loss = F.binary_cross_entropy_with_logits(logits, y_gradient)

            elif loss_type == 'tog':
                y_gradient = 2 * labels * torch.sigmoid(-2 * labels * bias)
                loss_q = F.binary_cross_entropy_with_logits(q_out, y_gradient)
                ref_logits = torch.sigmoid(q_pred) + bias
                y_gradient = 2 * labels * torch.sigmoid(-2 * labels * ref_logits)
                loss = F.binary_cross_entropy_with_logits(logits, y_gradient) + loss_q
                loss *= labels.size(1)
            
            elif loss_type == 'd_bias':
                loss = self.debias_loss_fn(None, logits, bias, labels)

            elif loss_type == 'q_bias':
                loss_q = F.binary_cross_entropy_with_logits(q_out, labels) * labels.size(1)
                ref_logits = torch.sigmoid(q_pred)
                loss = self.debias_loss_fn(None, logits, ref_logits, labels) + loss_q

            else:
                loss = self.debias_loss_fn(joint_repr, logits, bias, labels).mean(0)

        else:
            loss = None
        return logits, loss, atten[:,-1,:,:].sum(-1)

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
    attention = weight_norm(BiAttention(
            v_features=dataset.v_dim,
            q_features=q_emb.num_hid,
            mid_features=num_hid,
            glimpses=3,
            drop=0.5,), name='h_weight', dim=None)
    apply_attention = ApplyAttention(
            v_features=dataset.v_dim,
            q_features=q_emb.num_hid,
            mid_features=num_hid,
            glimpses=3,
            num_obj=36,
            drop=0.2,
        )
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    c_1=MLP(input_dim=q_emb.num_hid,dimensions=[1024,1024,dataset.num_ans_candidates])
    c_2=nn.Linear(dataset.num_ans_candidates,dataset.num_ans_candidates)

    return BaseModel(w_emb, q_emb, attention, q_net, v_net, classifier, c_1, c_2, apply_attention)