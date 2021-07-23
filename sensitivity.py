import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os
from dataset import Dictionary, VQAFeatureDataset
from torch.utils.data import DataLoader
import numpy as np
import click

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random

import base_model_block as base_model

from vqa_debias_loss_functions import *

def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=True,
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--mode', default="updn",
        choices=["updn", "q_debias","v_debias","q_v_debias"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none",'focal' , 'rubi', 'gradient'],
        help="Kind of ensemble loss to use")
    

    # Arguments from the original model, we leave this default, except we
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='analysis/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default='logs/baseline')
    parser.add_argument('--load_qid2score', type=str, default=None)
    args = parser.parse_args()
    return args

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def comput_attention_matching(att, hint_score):
    att_masked = (att > 0.2).float()

    # att_sort, att_ind = att.sort(1, descending=True)
    # v_ind = att_ind[:, :3]
    # att_masked = torch.zeros(*att.size()).cuda()
    # att_masked.scatter_(1, v_ind, 1)

    scores = (att_masked * hint_score)
    return scores

def evaluate(model, dataloader, qid2type):
    qid_to_prediction_scores = {}
    qid_to_attention_matching = {}
    score = 0.
    att_matching = 0.
    upper_bound = 0.

    right_att_matching = 0.
    right_count = 0
    wrong_count = 0

    wrong_att_matching = 0.

    for v, q, a, b, qids, hint_score, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=True).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        pred, _, atts = model(v, q, None, None, None, q_mask, loss_type = None)

        batch_scores = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_scores.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()

        att_scores = comput_attention_matching(atts.squeeze(-1).detach(), hint_score.cuda()).cpu().numpy().sum(1)
        att_matching += att_scores.sum()

        for qid, predict_score, att_score, hint_score in zip(qids, batch_scores, att_scores, hint_score):
            qid_to_prediction_scores[int(qid)] = float(predict_score)
            qid_to_attention_matching[int(qid)] = float(att_score)

            if predict_score >= 0.6:
                if hint_score.data.sum()>0.:
                    right_count += 1
                    right_att_matching += att_score
            else:
                if hint_score.sum()> 0:
                    wrong_att_matching += att_score
                    wrong_count += 1

            

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    att_matching = att_matching / len(dataloader.dataset)
    right_att_matching = right_att_matching / right_count
    wrong_att_matching = wrong_att_matching / wrong_count #(len(dataloader.dataset) - right_count)
    print(right_count)
    print(wrong_count)

    results = dict(
        qid_to_score = qid_to_prediction_scores,
        upper_bound=upper_bound,
        acc_score = score,
        q_id_to_att = qid_to_attention_matching,
        att_score = att_matching,
        right_att_score = right_att_matching,
        wrong_att_score = wrong_att_matching,
                 
        )
    return results

def main():
    args = parse_args()
    dataset=args.dataset
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)

        # else:
        #     os._exit(1)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))


    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                #   cache_image_features=args.cache_features)
                                cache_image_features=False)


    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    if dataset=='cpv1':
        model.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    # Add the loss_fn based our arguments
    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    elif args.debias=='focal':
        model.debias_loss_fn = Focal()
    elif args.debias=='boosting':
        model.debias_loss_fn = BoostFusion(eval_dset.num_ans_candidates)
    elif args.debias == 'rubi':
        pass
    else:
        raise RuntimeError(args.mode)


    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    ckpt = torch.load(os.path.join(args.load_checkpoint_path, 'model.pth'))
    if 'epoch' in ckpt:
        states_ = ckpt['model_state_dict']
    else:
        states_ = ckpt

    # model.load_state_dict(states_)

    model_dict = model.state_dict()
    ckpt = {k: v for k, v in states_.items() if k in model_dict}
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = args.batch_size

    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Start to test %s..." % args.load_checkpoint_path)
    model.train(False)
    results = evaluate(model, eval_loader, qid2type)

    eval_score = results["acc_score"]
    bound = results["upper_bound"]
    att_score = results["att_score"]
    right_att_score = results["right_att_score"]
    wrong_att_score = results["wrong_att_score"]
    qid_to_score = results["qid_to_score"]
    

    logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    logger.write('\tCGR: %.2f' % (100 * right_att_score))
    logger.write('\tCGW: %.2f' % (100 * wrong_att_score))
    logger.write('\tCGD: %.2f' % (100 * (right_att_score - wrong_att_score)))

    pickle.dump(qid_to_score, open(
            os.path.join(args.output,
                         'qid_to_score.pkl'),
            'wb'))

    if args.load_qid2score is not None:
        agree = 0.
        score_ref = pickle.load(open(os.path.join(args.load_qid2score, 'qid_to_score.pkl'), 'rb'))
        for k in qid_to_score.keys():
            if qid_to_score[k] <= score_ref[k]:
                agree += 1
        agree = agree / len(qid_to_score.keys())
        logger.write('\tagree: %.2f' % (100 * agree))


 

if __name__ == '__main__':
    main()    
