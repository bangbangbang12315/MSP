import os
import requests
import io 
import pkuseg
import json
import itertools
import logging
import argparse
import sentencepiece as spm
from rouge import Rouge
from tqdm import tqdm 
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk import word_tokenize, sent_tokenize 
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk import bigrams, FreqDist
from itertools import chain

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):

    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    # return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
    sentencelist = []
    for s in sentences:
        sentencelist.extend(s)
    return distinct_n_sentence_level(sentencelist,n)

def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\''
    C_pun = u'，。！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)

def ppl(textTest,train,n_gram=4):
    n = n_gram
    tokenized_text = [list(map(str.lower, sent)) 
                    for sent in train]
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    tokenized_text = [list(map(str.lower, sent)) 
                    for sent in textTest]
    test_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    model = Laplace(1) 
    model.fit(train_data, padded_sents)

    s = 0
    for i, test in enumerate(test_data):
        p = model.perplexity(test)
        s += p
    return s / (i + 1)
def rouge(candidate, reference):
    '''
    f:F1值  p：查准率  R：召回率
    a = ["i am a student from china"]  # 预测摘要 （可以是列表也可以是句子）
    b = ["i am student from school on japan"] #真实摘要
    '''
    rouge = Rouge()
    if len(candidate) == 0 or len(candidate) == 1:
        candidate.append('<UNK>')
    if len(reference) == 0:
        reference.append('<UNK>')
    rouge_score = rouge.get_scores(" ".join(candidate), " ".join(reference))
    return rouge_score[0]["rouge-1"]['r'], rouge_score[0]["rouge-2"]['r'], rouge_score[0]["rouge-l"]['r']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flag = 1 #0: no additional process; 1: seg cut; 2: bpe cut
    parser.add_argument('--test',default="test.txt", help="test file",type=str)
    parser.add_argument('--infer',default="results.txt", help="inference file",type=str)
    parser.add_argument('--dir_path',default="./evaluate/persona", help="dir path",type=str)
    parser.add_argument('--split_type',default=0, help="split type",type=int)
    args = parser.parse_args()
    
    flag = args.split_type
    if flag == 1:
        seg = pkuseg.pkuseg()
    if  flag == 2: 
        sp_dir_path = '../../data/Pdata/weibo'
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(sp_dir_path, 'spm_8000.model'))
    if not os.path.exists(os.path.join(args.dir_path, 'logging')):
        os.mkdir(os.path.join(args.dir_path, 'logging'))
    logging_fp = os.path.join(args.dir_path, 'logging', os.path.split(args.infer)[-1]  + '.log')
    log_formatter = logging.Formatter('%(message)s')
    log_handler = logging.FileHandler(logging_fp)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger('eval')
    logger.addHandler(log_handler)
    logger.setLevel(level=logging.INFO)
    # corpus_train = os.path.join(args.dir_path, 'infer', args.test)
    corpus_train = args.test
    text = []
    num = 0
    idx = 0
    # out_file = open(os.path.join(args.dir_path ,args.infer),'r')
    out_file = open(args.infer,'r')
    candidate = []
    bleu_score_all_1 = 0
    bleu_score_all_2 = 0
    bleu_score_all_3 = 0
    bleu_score_all_4 = 0
    rouge_score_all_1 = 0
    rouge_score_all_2 = 0
    rouge_score_all_l = 0
    train_sentence = []
    for line in out_file:
        if flag == 1:
            text = ''.join(line.strip().split(' '))
            text = E_trans_to_C(text)
            r = seg.cut(text)
        elif flag == 2:
            if args.dir_path == 'weibo':
                line = ''.join(line.strip().split(' '))
            r = sp.EncodeAsIds(line)
            r = [str(x) for x in r]
        else:
            r = line.strip().split(' ')
        candidate.append(r)
    with open(corpus_train,'r') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            reference = []
            data = line
            resps_num = len(data)
            for resp in data.strip().split('\t'):
                if flag == 2:
                    if 'Weibo' in args.dir_path:
                        resp = ''.join(resp.split(' '))
                    word_idx = sp.EncodeAsIds(resp)
                    reference.append([str(x) for x  in word_idx])
                    train_sentence.append([str(x) for x  in word_idx])
                else:
                    if 'Weibo' in args.dir_path and flag == 1:
                        resp = E_trans_to_C(''.join(resp.split(' ')))
                        resp = seg.cut(resp)
                    else:
                        resp = resp.split(' ')
                    reference.append(resp)
                    # train_sentence.append(resp.split(' '))
            bleu_score_1 = sentence_bleu(reference, candidate[idx],weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = sentence_bleu(reference, candidate[idx],weights=(0.5, 0.5, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = sentence_bleu(reference, candidate[idx],weights=(0.33, 0.33, 0.33, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = sentence_bleu(reference, candidate[idx],weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score_all_4 += bleu_score_4
            rouge_score_1, rouge_score_2, rouge_score_l = rouge(candidate[idx], reference[0])
            rouge_score_all_1 += rouge_score_1
            rouge_score_all_2 += rouge_score_2
            rouge_score_all_l += rouge_score_l
            num += 1
    # ppl_score_1 = ppl(candidate,train_sentence,1)
    # ppl_score_2 = ppl(candidate,train_sentence,2)
    # print(ppl_score_1, ppl_score_2)
    distinct_score_1 = distinct_n_corpus_level(candidate,1)
    distinct_score_2 = distinct_n_corpus_level(candidate,2)
    logger.info('BLEU-1:%f, BLEU-2:%f,BLEU-3:%f,BLEU-4:%f,DISTINCT-1:%f,DISTINCT-2:%f, ROUGE-1:%f, ROUGE-2:%f, ROUGE-L:%f',
        bleu_score_all_1 / num, bleu_score_all_2 / num, bleu_score_all_3 / num, bleu_score_all_4 / num,
        distinct_score_1,distinct_score_2, rouge_score_all_1 / num, rouge_score_all_2/ num, rouge_score_all_l / num)
    out_file.close()
