#!/usr/bin/env bash

dataset="$1"
datadir="$2"
task="$3"


# -u to check bounded args, -e to exit when error
#set -u
set -e

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

if [ ! -n "$task" ]; then
    task=single
fi

type=(train val test)

echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
python script/createVoc.py --dataset $dataset --data_path $datadir/train.label.jsonl

echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
python script/lowTFIDFWords.py --dataset $dataset --data_path $datadir/train.label.jsonl

echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in ${type[*]}
    do
        python script/calw2sTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
    done

if [ "$task" == "multi" ]; then
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
    for i in ${type[*]}
        do
            python script/calw2dTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
        done
fi

echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"


