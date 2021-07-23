import torch
import torch.nn as nn
from attention import Attention, NewAttention, SelfAttention, Block
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
    def __init__(self, w_emb, q_emb, v_att, q_att, q_net, fusion, v_net, classifier, c_1,c_2):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_att = q_att
        self.q_net = q_net
        self.fusion = fusion
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        self.loss_ref = LearnedMixin(0.36)
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.c_1=c_1
        self.c_2=c_2
        self.vision_lin = torch.nn.Linear(1024, 1)
        self.question_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, q, labels, bias, v_mask, weights, loss_type = None):
        """Forward
 
        v: [batch, num_objs, obj_dim] 
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        # q_out = self.q_net_2(q_emb)  

        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]      

        # q_repr = self.q_net(q_emb)
        # v_repr = self.v_net(v_emb)      

        # joint_repr = v_repr * q_repr
        # batch, k, _ = v.size()
        # q_expand = q_emb.unsqueeze(1).repeat(1, k, 1)

        # q_expand = q_expand.contiguous().view(batch * k, -1)

        # v_expand = v.contiguous().view(batch * k, -1)
        # mm = self.fusion([q_expand, v_expand]).view(batch,  k, -1)
        # joint_repr, _ = torch.max(mm, 1)

        joint_repr = self.fusion([q_emb, v_emb])

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
    fusion = Block(input_dims=[q_emb.num_hid, dataset.v_dim], output_dim=num_hid)
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    classifier_v = SimpleClassifier(
        num_hid, num_hid, dataset.num_ans_candidates, 0.5)
    classifier_q = SimpleClassifier(
        num_hid, num_hid, dataset.num_ans_candidates, 0.5)

    c_1=MLP(input_dim=q_emb.num_hid,dimensions=[1024,1024,dataset.num_ans_candidates])
    c_2=nn.Linear(dataset.num_ans_candidates,dataset.num_ans_candidates)

    return BaseModel(w_emb, q_emb, v_att, q_att, q_net, fusion, v_net, classifier, c_1, c_2)