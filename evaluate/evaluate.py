import os
from nltk import data
import pkuseg
import json
import logging
import argparse
import math
import gensim
import pickle
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.translate.bleu_score import sentence_bleu
from itertools import chain
class Evaluator():
    def __init__(self, dataformat) -> None:
        if dataformat == 'weibo':
            dbFile =  open("zh.word.pkl","rb")
            stopwords = './chinese.txt'
        else:
            dbFile =  open("en.word.pkl","rb")
            stopwords = './english.txt'
        self.global_idf_dict = pickle.load(dbFile)
        self.stop_words_set = self.load_stopwords(stopwords)
        dbFile.close()

    def load_stopwords(self, fp):
        stopwords = []
        with open(fp, 'r') as fwords:
            for word in fwords:
                stopwords.append(word.strip())
        return stopwords

    def BLUE(self, reference, candidate, weights):
        return sentence_bleu(reference, candidate, weights)

    def Rouge(self, candidate, reference):
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

    def P_Cover(self, inference, historys):
        s_sum = 0
        line_cnt = 0
        for result, history in zip(inference, historys):
            s_list = []
            for h in history:
                s = self.cal_s_for_each_history(result, h)
                s_list.append(s)
            s_max = max(s_list)
            print(s_max)
            s_sum += s_max
            line_cnt += 1
        return (s_sum + 0.0) / line_cnt

    def P_F1(self, inference, historys):
        score = 0
        cnt = 0
        for result, history in zip(inference, historys):
            score += self.cal_f1(result, history)
            cnt += 1
        return score / cnt

    def cal_f1(self, result, history):
        h_all = []
        for i, h in enumerate(history):
            h =  ' '.join(h).replace("<PAD>","").replace("<EOS>","").replace("<SOS>","").split()
            h_all += h
        h_set = set(h_all) - set(self.stop_words_set)
        r_set = set(result) - set(self.stop_words_set)
        # h_set = set(h_all)
        # r_set = set(result)
        if len(h_set) == 0 or len(r_set) == 0:
            p, r = 0, 0
        else:
            p = len(h_set & r_set) / len(r_set)
            r = len(h_set & r_set) / len(h_set)
        # print(p,r)
        if p == r == 0:
            return 0
        return (2 * p * r) / (p + r)

    def Distinct(self, sentences, n):
        # return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
        sentencelist = []
        for s in sentences:
            sentencelist.extend(s)
        return self.distinct_n_sentence_level(sentencelist,n)

    def PPL(self, textTest,train,n_gram=4):
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

    def pad_sequence(self, sequence, n, pad_left=False, pad_right=False,
                    left_pad_symbol=None, right_pad_symbol=None):

        sequence = iter(sequence)
        if pad_left:
            sequence = chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence

    def ngrams(self, sequence, n, pad_left=False, pad_right=False,
            left_pad_symbol=None, right_pad_symbol=None):

        sequence = self.pad_sequence(sequence, n, pad_left, pad_right,
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

    def distinct_n_sentence_level(self, sentence, n):
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = set(self.ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)

    def E_trans_to_C(self, string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
        return string.translate(table)

    def preprocess_result(self, filepath):
        f = open(filepath)
        test_path = "test.eval"
        infer_path = "infer.eval"
        f1 = open(test_path,"w")
        f2 = open(infer_path,"w")
        for line in f:
            r = json.loads(line.strip())
            p = r['post']
            a = r['answer']
            res = r['result']

            a_str = ' '.join(a)
            r_str = ' '.join(res)
            a_str = a_str.replace("<SOS>","").replace("<EOS>","").replace("<PAD>","")
            r_str = r_str.replace("<SOS>","").replace("<EOS>","").replace("<PAD>","")
            if len(r_str.strip()) == 0:
                continue
            if len(a_str.strip()) == 0:
                continue
            f1.write(a_str+"\n")
            f2.write(r_str+"\n")
        f.close()
        f1.close()
        f2.close()
        return test_path, infer_path


    def cal_vector_extrema(self, x, y, dic):
        # x and y are the list of the words
        # dic is the gensim model which holds 300 the google news word2ved model
        def vecterize(p):
            vectors = []
            for w in p:
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        vec_x = np.max(x, axis=0)
        vec_y = np.max(y, axis=0)
        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
        zero_list = np.zeros(len(vec_x))
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)
        res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return cos

    def cal_embedding_average(self, x, y, dic):
        # x and y are the list of the words
        def vecterize(p):
            vectors = []
            for w in p:
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)

        vec_x = np.array([0 for _ in range(len(x[0]))])
        for x_v in x:
            x_v = np.array(x_v)
            vec_x = np.add(x_v, vec_x)
        vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))

        vec_y = np.array([0 for _ in range(len(y[0]))])
        #print(len(vec_y))
        for y_v in y:
            y_v = np.array(y_v)
            vec_y = np.add(y_v, vec_y)
        vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))

        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"

        zero_list = np.array([0 for _ in range(len(vec_x))])
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)

        vec_x = np.mat(vec_x)
        vec_y = np.mat(vec_y)
        num = float(vec_x * vec_y.T)
        denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
        cos = num / denom

        # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return cos

    def cal_greedy_matching(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    vectors.append(dic[w.lower()])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)

        len_x = len(x)
        len_y = len(y)

        cosine = []
        sum_x = 0

        for x_v in x:
            for y_v in y:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(x_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if x_v.all() == y_v.all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)
            if cosine:
                sum_x += max(cosine)
                cosine = []

        sum_x = sum_x / len_x
        cosine = []
        sum_y = 0
        for y_v in y:
            for x_v in x:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(y_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if (x_v == y_v).all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)

            if cosine:
                sum_y += max(cosine)
                cosine = []

        sum_y = sum_y / len_y
        score = (sum_x + sum_y) / 2
        return score

    def cal_greedy_matching_matrix(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)     # [x, 300]
        y = vecterize(y)     # [y, 300]

        len_x = len(x)
        len_y = len(y)

        matrix = np.dot(x, y.T)    # [x, y]
        matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
        matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]

        x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
        y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]

        return (x_matrix_max + y_matrix_max) / 2

    def cal_s_for_each_history(self, r, h):
        # print(r,h)
        # return 1
        if len(r) == 0:
            return 0
        c = 0
        has_c = {}
        for w in r:
            if w in h and w not in has_c:
                # c += idf_dict[w]
                if w in self.global_idf_dict:
                    c += self.global_idf_dict[w]
                    has_c[w] = 1
        return c / len(r)

def cut(cut_flag, line, seg=None):
    if cut_flag == 0:
        return line.split(' ')
    else:
        return seg.cut(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',default="test.txt", help="test file",type=str)
    parser.add_argument('--infer',default="results.txt", help="inference file",type=str)
    parser.add_argument('--history',default="history.txt", help="history file",type=str)
    parser.add_argument('--dir_path',default="./persona", type=str)
    parser.add_argument('--dataformat',default="weibo",type=str)
    parser.add_argument('--cut_flag',default=0, help="#0: no additional process; 1: seg cut; 2: bpe cut", type=int)
    args = parser.parse_args()
    evaluate = Evaluator(args.dataformat)
    seg = None
    if args.cut_flag == 1:
        seg = pkuseg.pkuseg()
    if not os.path.exists(os.path.join(args.dir_path, 'logging')):
        os.mkdir(os.path.join(args.dir_path, 'logging'))
    logging_fp = os.path.join(args.dir_path, 'logging', args.infer + '.log')
    log_formatter = logging.Formatter('%(message)s')
    log_handler = logging.FileHandler(logging_fp)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger('eval')
    logger.addHandler(log_handler)
    logger.setLevel(level=logging.INFO)
    reference_data = os.path.join(args.dir_path, args.test)
    text = []
    num = 0
    idx = 0
    inference_data = open(os.path.join(args.dir_path ,args.infer),'r')
    infer = []
    bleu_score_all_1 = 0
    bleu_score_all_2 = 0
    bleu_score_all_3 = 0
    bleu_score_all_4 = 0
    rouge_score_all_1 = 0
    rouge_score_all_2 = 0
    rouge_score_all_l = 0
    for line in inference_data:
        if args.cut_flag == 1:
            text = ''.join(line.strip().split(' '))
            text = evaluate.E_trans_to_C(text)
        else:
            text = line.strip()
        r = cut(args.cut_flag, text, seg)
        infer.append(r)
    print('Caculating BLUE and Rouge....')
    with open(reference_data,'r') as f_refer:
        for idx, line in tqdm(enumerate(f_refer.readlines())):
            reference = []
            resps_num = len(line)
            for resp in line.strip().split('\t'):
                    if args.dataformat == 'weibo' and args.cut_flag == 1:
                        resp = evaluate.E_trans_to_C(''.join(resp.split(' ')))
                    resp = cut(args.cut_flag, resp, seg)
                    reference.append(resp)
            bleu_score_1 = evaluate.BLUE(reference, infer[idx],weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = evaluate.BLUE(reference, infer[idx],weights=(0.5, 0.5, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = evaluate.BLUE(reference, infer[idx],weights=(0.33, 0.33, 0.33, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = evaluate.BLUE(reference, infer[idx],weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score_all_4 += bleu_score_4
            rouge_score_1, rouge_score_2, rouge_score_l = evaluate.Rouge(infer[idx], reference[0])
            rouge_score_all_1 += rouge_score_1
            rouge_score_all_2 += rouge_score_2
            rouge_score_all_l += rouge_score_l
            num += 1
    # ppl_score_1 = ppl(candidate,train_sentence,1)
    # ppl_score_2 = ppl(candidate,train_sentence,2)
    # print(ppl_score_1, ppl_score_2)
    print('Caculating Distinct....')
    distinct_score_1 = evaluate.Distinct(infer,1)
    distinct_score_2 = evaluate.Distinct(infer,2)
    inference_data.close()
    print('[!] load the word2vector by gensim over')
    if args.dataformat == 'reddit':
        dic = gensim.models.KeyedVectors.load_word2vec_format('./fasttext.300d.txt.withheader', binary=False)
    else:
        dic = gensim.models.KeyedVectors.load_word2vec_format('./sgns.weibo.bigram-char-withheader', binary=False)

    print('Caculating Emb....')
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    no_save = 0
    for rr, cc in tqdm(list(zip(reference, infer))):
        ea_sum_ = evaluate.cal_embedding_average(rr, cc, dic)
        vx_sum_ = evaluate.cal_vector_extrema(rr, cc, dic)
        gm_sum += evaluate.cal_greedy_matching_matrix(rr, cc, dic)
        # gm_sum += cal_greedy_matching(rr, cc, dic)
        if ea_sum_ != 1 and vx_sum_ != 1:
            ea_sum += ea_sum_
            vx_sum += vx_sum_
            counterp += 1
        else:
            no_save += 1

    # print(f'EA: {round(ea_sum / counterp, 4)}')
    # print(f'VX: {round(vx_sum / counterp, 4)}')
    # print(f'GM: {round(gm_sum / counterp, 4)}')
    historys = []
    print('Caculating P_Cover and P_F1....')
    with open(os.path.join(args.dir_path, args.history), 'r') as fhis:
        for line in tqdm(fhis):
            line = line.strip().split('\t')
            historys.append(list(map(lambda x: cut(args.cut_flag, evaluate.E_trans_to_C(x), seg),line)))
    print(infer[0], historys[0])
    pcover = evaluate.P_Cover(infer, historys)
    pf1 = evaluate.P_F1(infer, historys)

    logger.info('BLEU-1:%f, BLEU-2:%f,BLEU-3:%f,BLEU-4:%f,DISTINCT-1:%f,DISTINCT-2:%f, ROUGE-1:%f, ROUGE-2:%f, ROUGE-L:%f, EA:%f, VX:%f, GM:%f, PCover:%f, PF1:%f',
        bleu_score_all_1 / num, bleu_score_all_2 / num, bleu_score_all_3 / num, bleu_score_all_4 / num,
        distinct_score_1,distinct_score_2, rouge_score_all_1 / num, rouge_score_all_2/ num, rouge_score_all_l / num,
        ea_sum / counterp, vx_sum / counterp, gm_sum / counterp, pcover, pf1)