import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
# import base_model_block as base_model
# import base_model_ban as base_model
# import base_model_ab as base_model
# import base_model_v_only as base_model
# import base_model_sfce as base_model

# from train_ab import train
from train_GGE import train
import utils
import click

from vqa_debias_loss_functions import *


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=False,
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
        '--mode', default="base",
        choices=["updn", "base", "v_inverse","v_debias", "inv_sup", "gge_iter", "gge_tog", "gge_d_bias", "gge_q_bias"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none","focal", "gradient", "gradient_sfce"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--topq', type=int,default=1,
        choices=[1,2,3],
        help="num of words to be masked in questio")
    parser.add_argument(
        '--keep_qtype', default=True,
        help="keep qtype or not")
    parser.add_argument(
        '--topv', type=int,default=1,
        choices=[1,3,5,-1],
        help="num of object bbox to be masked in image")
    parser.add_argument(
        '--top_hint',type=int, default=9,
        choices=[9,18,27,36],
        help="num of hint")
    parser.add_argument(
        '--qvp', type=int,default=0,
        choices=[0,1,2,3,4,5,6,7,8,9,10],
        help="ratio of q_bias and v_bias")
    parser.add_argument(
        '--eval_each_epoch', default=True,
        help="Evaluate every epoch, instead of at the end")

    # Arguments from the original model, we leave this default, except we
    # set --epochs to 30 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args

def get_bias(train_dset,eval_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]



def main():
    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)

        else:
            if args.load_checkpoint_path is None:
                os._exit(1)



    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                #    cache_image_features=args.cache_features)
                                cache_image_features=False)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                #   cache_image_features=args.cache_features)
                                cache_image_features=False)

    get_bias(train_dset,eval_dset)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
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
    elif args.debias=='gradient':
        model.debias_loss_fn = GreedyGradient()
    elif args.debias=='gradient_sfce':
        model.debias_loss_fn = GreedyGradient_softmaxCE()
    else:
        raise RuntimeError(args.mode)


    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    if args.load_checkpoint_path is not None:
        ckpt = torch.load(os.path.join('logs', args.load_checkpoint_path, 'model.pth'))
        # states_ = ckpt
        # model.load_state_dict(states_)
        model_dict = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(model_dict)
        model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    train(model, train_loader, eval_loader, args,qid2type)

if __name__ == '__main__':
    main()






