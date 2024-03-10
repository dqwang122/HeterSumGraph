#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import datetime
import os
import shutil
import time
import random

import dgl
import numpy as np
import torch
from rouge import Rouge

from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *

_DEBUG_FLAG_ = False


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, hps):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """

    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def run_training(model, train_loader, valid_loader, valset, hps, train_dir):
    '''  Repeatedly runs training iterations, logging loss to screen and log files
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return: 
    '''
    logger.info("[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)


    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            iter_start_time = time.time()
            # if i > 10:
            #     break
            model.train()

            if hps.cuda:
                G = G.to(torch.device("cuda"))
            graph_device = G.device
            outputs = model.forward(G)  # [n_snodes, 2]
            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
            G.nodes[snode_id].data["loss"] = criterion(outputs, label).unsqueeze(-1)  # [n_nodes, 1]
            loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
            loss = loss.mean()

            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        # logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            optimizer.zero_grad()
            loss.backward()
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            optimizer.step()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(i, (time.time() - iter_start_time),float(train_loss / 100)))
                train_loss = 0.0

        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo)

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            return


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo):
    ''' 
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return: 
    '''
    logger.info("[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(hps.save_root, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model.eval()

    iter_start_time = time.time()

    with torch.no_grad():
        tester = SLTester(model, hps.m)
        for i, (G, index) in enumerate(loader):
            if hps.cuda:
                G.to(torch.device(0))
            tester.evaluation(G, index, valset)

    running_avg_loss = tester.running_avg_loss

    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | ' .format((time.time() - iter_start_time), float(running_avg_loss)))

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    F = tester.labelMetric

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (saveNo % 3))  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < F:
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel')  # this is where checkpoints of best models are saved
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F),
                        float(best_F), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s', float(F),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo


def main():
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM',help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM',help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=666, help='set the random seed [default: 666]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=300, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100,help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()
    
    # set the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root


    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    hps = args
    logger.info(hps)

    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,collate_fn=graph_collate_fn)
        del dataset
        valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)
    elif hps.model == "HDSG":
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
        train_w2d_path = os.path.join(args.cache_dir, "train.w2d.tfidf.jsonl")
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,collate_fn=graph_collate_fn)
        del dataset
        val_w2d_path = os.path.join(args.cache_dir, "val.w2d.tfidf.jsonl")
        valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,collate_fn=graph_collate_fn, num_workers=32)  # Shuffle Must be False for ROUGE evaluation
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")


    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    setup_training(model, train_loader, valid_loader, valid_dataset, hps)


if __name__ == '__main__':
    main()
