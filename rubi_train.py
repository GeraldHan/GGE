import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader,args,qid2type):
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0



    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp



    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):

            total_step += 1


            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda()
            notype_mask=Variable(notype_mask).float().cuda()
            #########################################

            if mode=='updn':
                #plain model
                pred, loss,_ = model(v, q, a, b, None)
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

            elif mode=='lmh_rubi':
                pred, loss,_ = model(v, q, a, b, None, loss_type = 'lmh_rubi')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                
                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score


        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, b, qids, _, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _,_ = model(v, q, None, None, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')


    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
