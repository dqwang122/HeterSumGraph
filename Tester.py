import torch
import dgl

import os
from tools.utils import eval_label
from tools.logger import *

class TestPipLine():
    def __init__(self, model, m, test_dir, limited):
        """
            :param model: the model
            :param m: the number of sentence to select
            :param test_dir: for saving decode files
            :param limited: for limited Recall evaluation
        """
        self.model = model
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []

        self.batch_number = 0
        self.running_loss = 0
        self.example_num = 0
        self.total_sentence_num = 0

        self._hyps = []
        self._refer = []

    def evaluation(self, G, index, valset):
        pass

    def getMetric(self):
        pass

    def SaveDecodeFile(self):
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        log_dir = os.path.join(self.test_dir, nowTime)
        with open(log_dir, "wb") as resfile:
            for i in range(self.rougePairNum):
                resfile.write(b"[Reference]\t")
                resfile.write(self._refer[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(self._hyps[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")

    @property
    def running_avg_loss(self):
        return self.running_loss / self.batch_number

    @property
    def rougePairNum(self):
        return len(self._hyps)

    @property
    def hyps(self):
        if self.limited:
            hlist = []
            for i in range(self.rougePairNum):
                k = len(self._refer[i].split(" "))
                lh = " ".join(self._hyps[i].split(" ")[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        return self._refer

    @property
    def extractLabel(self):
        return self.extracts

    
class SLTester(TestPipLine):
    def __init__(self, model, m, test_dir=None, limited=False, blocking_win=3):
        super().__init__(model, m, test_dir, limited)
        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0
        self._F = 0
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.blocking_win = blocking_win

    def evaluation(self, G, index, dataset, blocking=False):
        """
            :param G: the model
            :param index: list, example id
            :param dataset: dataset which includes text and summary
            :param blocking: bool, for n-gram blocking
        """
        self.batch_number += 1
        outputs = self.model.forward(G)
        # logger.debug(outputs)
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)            # [n_nodes]
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label).unsqueeze(-1)    # [n_nodes, 1]
        loss = dgl.sum_nodes(G, "loss")    # [batch_size, 1]
        loss = loss.mean()
        self.running_loss += float(loss.data)

        G.nodes[snode_id].data["p"] = outputs
        glist = dgl.unbatch(G)
        for j in range(len(glist)):
            idx = index[j]
            example = dataset.get_example(idx)
            original_article_sents = example.original_article_sents
            sent_max_number = len(original_article_sents)
            refer = example.original_abstract

            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            N = len(snode_id)
            p_sent = g.ndata["p"][snode_id]
            p_sent = p_sent.view(-1, 2)   # [node, 2]
            label = g.ndata["label"][snode_id].sum(-1).squeeze().cpu()    # [n_node]
            if self.m == 0:
                prediction = p_sent.max(1)[1] # [node]
                pred_idx = torch.arange(N)[prediction!=0].long()
            else:
                if blocking:
                    pred_idx = self.ngram_blocking(original_article_sents, p_sent[:,1], self.blocking_win, min(self.m, N))
                else:
                    # print(p_sent.size())
                    topk, pred_idx = torch.topk(p_sent[:,1], min(self.m, N))
                prediction = torch.zeros(N).long()
                prediction[pred_idx] = 1
            self.extracts.append(pred_idx.tolist())

            self.pred += prediction.sum()
            self.true += label.sum()

            self.match_true += ((prediction == label) & (prediction == 1)).sum()
            self.match += (prediction == label).sum()
            self.total_sentence_num += N
            self.example_num += 1
            hyps = "\n".join(original_article_sents[id] for id in pred_idx if id < sent_max_number)

            self._hyps.append(hyps)
            self._refer.append(refer)

    def getMetric(self):
        logger.info("[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d",
                    self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        logger.info(
            "[INFO] The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu, self._precision, self._recall, self._F)


    def ngram_blocking(self, sents, p_sent, n_win, k):
        """
        
        :param p_sent: [sent_num, 1]
        :param n_win: int, n_win=2,3,4...
        :return: 
        """
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        S = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []
            for i in range(len(pieces) - n_win):
                ngram = " ".join(pieces[i : (i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)
            if overlap_flag == 0:
                S.append(idx)
                ngram_list.extend(sent_ngram)
                if len(S) >= k:
                    break
        S = torch.LongTensor(S)
        # print(sorted_idx, S)
        return S


    @property
    def labelMetric(self):
        return self._F

