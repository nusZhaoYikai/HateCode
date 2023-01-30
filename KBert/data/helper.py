"""
生成训练集、验证集、测试集的pos、pos_tag文件
以及pos、pos_tag的词典文件
"""
import json
import os

import pandas as pd
import spacy
from tqdm import tqdm
from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

nlp = spacy.load('en_core_web_sm')

data_dir = ""
train_file = "train_data.csv"
dev_file = "dev_data.csv"
test_file = "test_data.csv"
max_pos = 0
postag_vocab = {"[PAD]"}


def process(file_path):
    global max_pos, postag_vocab
    mode = file_path.replace("_data.csv", "")
    print("Processing {} data...".format(mode))
    data = pd.read_csv(file_path, header=0)
    # 去除数据中tweet为空的行
    # data = data[data['tweet'].notnull()]
    data = data.values.tolist()
    pos = dict()
    postag = dict()
    pbar = tqdm(enumerate(data), total=len(data))
    for i, item in pbar:
        text, label = item
        try:
            doc = nlp(text)
            postag[str(i)] = [token.tag_ for token in doc]
            pos[str(i)] = [i for i in range(len(doc))]
            max_pos = max(max_pos, len(doc) - 1)
            postag_vocab.update(postag[str(i)])
        except Exception as e:
            print(e)
            print(text)
            print(label)
            continue
        pbar.set_description("Processing %s" % mode)
        pbar.update(1)

    # 保存pos、pos_tag文件
    with open(os.path.join(data_dir, f"{mode}_pos.json"), "w", encoding="utf-8") as f:
        json.dump(pos, f, ensure_ascii=False)
    with open(os.path.join(data_dir, f"{mode}_postag.json"), "w", encoding="utf-8") as f:
        json.dump(postag, f, ensure_ascii=False)


process(train_file)
process(dev_file)
process(test_file)

# 根据max_pos生成pos词典文件
with open(os.path.join(data_dir, "pos_vocab.txt"), "w", encoding="utf-8") as f:
    for i in range(max_pos + 1):
        if i == max_pos:
            f.write(str(i))
        else:
            f.write(str(i) + "\n")

# 根据postag_vocab生成postag词典文件
with open(os.path.join(data_dir, "postag_vocab.txt"), "w", encoding="utf-8") as f:
    for i, tag in enumerate(postag_vocab):
        if i == len(postag_vocab) - 1:
            f.write(tag)
        else:
            f.write(tag + "\n")
