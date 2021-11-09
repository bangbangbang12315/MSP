import json
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizerFast
import argparse
import pandas as pd
import pickle
import jieba.analyse
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import logging
import numpy as np
import os

def split_test(test_fp, test_ans, test_post, test_ref):
    ## todo
    with open(test_fp, 'rb') as fp:
        data = fp.read().decode("utf-8")
    test_data = data.split('\n')
    if test_ref != None:
        with open(test_post, 'w') as fsrc, open(test_ans, 'w') as ftgt, open(test_ref, 'w') as fref:
            for line in tqdm(test_data):
                line_list = line.split('###')
                post, resp, pref, sref, oref = line_list[0], line_list[1], line_list[2], line_list[3], line_list[4]
                reference = '\t'.join(ref)
                fsrc.write(post+'\n')
                ftgt.write(resp+'\n')
                fref.write(reference+'\n')
    else:
        with open(test_post, 'w') as fsrc, open(test_ans, 'w') as ftgt:
            for line in tqdm(test_data):
                line_list = line.split('\n')
                if len(line_list) != 2:
                    continue
                post, resp = line_list[0], line_list[1]
                fsrc.write(post+'\n')
                ftgt.write(resp+'\n')

def delete_blank(sentences):
    if isinstance(sentences, list):
        ans = []
        for sentence in sentences:
            ans += delete_blank(sentence)
        return ans
    else:
        return [''.join(sentences.split(' '))]

def merge_data(post_fp, resp_fp, refer_fp, train_fp, test_fp):
    cnt = 0
    if test_fp != None:
        te_fp = open(test_fp, 'w')
    if refer_fp != None:
        with open(post_fp, 'r') as p_fp, open(resp_fp, 'r') as r_fp, open(refer_fp, 'r') as re_fp, open(train_fp, 'w') as tr_fp:
            for post, resp, refer in tqdm(zip(p_fp, r_fp, re_fp)):
                cnt += 1
                tgt = '\n'.join(delete_blank(refer.strip().split('\t'))) + '\n' + '\n'.join(delete_blank(post.strip())) + '\n' + '\n'.join(delete_blank(resp.strip())) +'\n'
                if cnt % 100 == 0:
                    if test_fp != None:
                        te_fp.write(tgt + '\n')
                    else:
                        tr_fp.write(tgt + '\n')
                else:
                    tr_fp.write(tgt + '\n')
    else:
        with open(post_fp, 'r') as p_fp, open(resp_fp, 'r') as r_fp, open(train_fp, 'w') as tr_fp:
            for post, resp in tqdm(zip(p_fp, r_fp)):
                cnt += 1
                tgt = '\n'.join(delete_blank(post.strip())) + '\n' + '\n'.join(delete_blank(resp.strip())) +'\n'
                if cnt % 100 == 0:
                    if test_fp != None:
                        te_fp.write(tgt + '\n')
                    else:
                        tr_fp.write(tgt + '\n')
                else:
                    tr_fp.write(tgt + '\n')

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger            

def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='pretrained/gpt2-chinese-cluecorpussmall/vocab.txt', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='ref/preprocess.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--train_path', default='ref/test/test.txt', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--save_path', default='ref/test/test.pkl', type=str, required=False, help='tokenize的训练数据集')
    parser.add_argument('--is_test', action='store_true', help='是否在做test的tokenize')
    args = parser.parse_args()
    # 初始化日志对象
    logger = create_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    logger.info("preprocessing data,data path:{}, save path:{}".format(args.train_path, args.save_path))

    # 读取训练数据集
    with open(args.train_path, 'rb') as f:
        data = f.read().decode("utf-8")

    # 需要区分linux和windows环境下的换行符
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in dataset".format(len(train_data)))

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    with open(args.save_path, "w", encoding="utf-8") as f:
        for index, dialogue in enumerate(tqdm(train_data)):
            # if "\r\n" in data:
            #     utterances = dialogue.split("\r\n")
            # else:
            utterances = dialogue.split("\n")
            if len(utterances) <= 1:
                continue
            if args.is_test:
                utterances = utterances[:-1]
            input_ids = [cls_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                input_ids += tokenizer.encode(utterance, add_special_tokens=False)
                input_ids.append(sep_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            dialogue_len.append(len(input_ids))
            dialogue_list.append(input_ids)
    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)
    with open(args.save_path, "wb") as f:
        pickle.dump(dialogue_list, f)
    logger.info("finish preprocessing data,the result is stored in {}".format(args.save_path))
    logger.info("mean of dialogue len:{},median of dialogue len:{},max len:{}".format(len_mean, len_median, len_max))


if __name__ == '__main__':
    # post_fp = './ref/test/post.txt'
    # resp_fp = './ref/test/resp.txt'
    # refer_fp = './ref/test/refer.txt'
    # # train_fp = './ref/train.txt'
    # test_fp = './ref/test/test.txt'
    # merge_data(post_fp, resp_fp, None, test_fp, None)

    test_fp = './ref/Selected_Weibo/test/test.txt'
    test_ans = './evaluate/Selected_Weibo/infer/test_ans.txt'
    test_post = './evaluate/Selected_Weibo/infer/test_post.txt'
    test_ref = './evaluate/ref/infer/test_ref.txt'
    split_test(test_fp, test_ans, test_post, None)

    preprocess()
    # getConstrativeDataset()

