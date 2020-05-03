import os
import json
import nltk
import random
import argparse


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def PrintInformation(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "vocab")
    print("Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    text = []
    summary = []
    allword = []
    cnt = 0
    with open(args.data_path, encoding='utf8') as f:
        for line in f:
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents = catDoc(e["text"])
            else:
                sents = e["text"]
            text = " ".join(sents)
            summary = " ".join(e["summary"])
            allword.extend(text.split())
            allword.extend(summary.split())
            cnt += 1
    print("Training set of dataset has %d example" % cnt)

    fdist1 = nltk.FreqDist(allword)

    fout = open(saveFile, "w")
    keys = fdist1.most_common()
    for key, val in keys:
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            # print(repr(e))
            # print(key, val)
            continue

    fout.close()

    allcnt = fdist1.N() # 788,159,121
    allset = fdist1.B() # 5,153,669
    print("All appearance %d, unique word %d" % (allcnt, allset))

    PrintInformation(keys, allcnt)
