from collections import defaultdict
import os
from io import SEEK_CUR
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


class DialoDataset(Dataset):
    def __init__(self, train_data_dir=None, valid_data_dir=None, test_data_dir=None, train=False, vocab_path="pretrained/gpt2-chinese-cluecorpussmall/vocab.txt", max_length=512,max_candidate_num=14):
        self.tok = BertTokenizer(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
        self.max_length = max_length
        self.max_candidate_num = max_candidate_num
        self.train = train
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.test_data_dir = test_data_dir
        if self.train:
            self.data = self.load_data(train_data_dir)
        else:
            if valid_data_dir != None:
                self.data = self.load_data(valid_data_dir)
            else:
                self.data = self.load_data(test_data_dir)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        tokenized_line = defaultdict(list)
        for k, v in line.items():
            if k != 'ref':
                if k == 'input_ids':
                    max_len = self.max_length
                else:
                    max_len = self.max_length // 2
                tokenized_line[k] = self.tok.encode(
                                        v,
                                        max_length=max_len,
                                        truncation=True,
                                        return_tensors="pt"
                                    ).squeeze()
            else:
                for candidate in v:
                    tokenized_line[k].append(self.tok.encode(
                                        candidate,
                                        max_length=self.max_length // 16,
                                        truncation=True,
                                        padding="max_length",
                                        return_tensors="pt"))
                tokenized_line[k] = torch.cat(tokenized_line[k],dim=0)
        
        return tokenized_line
    
    def load_data(self, data_dir):
        '''
        line: post###resp###ref
        '''
        data = []
        pad_ref = [self.tok.pad_token_id] * (self.max_length // 16)
        with open(data_dir, 'r') as fsrc:
            for sub in tqdm(fsrc, desc='Load Dataset'):
                post, resp, pref, sref, tref = sub.strip().split('###')
                # naive:
                sub_dict = {'post': ''.join(post.strip().split(' ')),
                            'resp': ''.join(resp.strip().split(' ')),
                            'input_ids': ''.join(post.strip().split(' ')) + '[SEP]' + ''.join(resp.strip().split(' ')),
                            'ref': pref.split('\t') + sref.split('\t') + tref.split('\t')}

                while len(sub_dict['ref']) < self.max_candidate_num:
                    sub_dict['ref'].append(pad_ref)
                sub_dict['ref'] = sub_dict['ref'][-self.max_candidate_num:]
                data.append(sub_dict)
        return data
