import importlib
import inspect

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
import random
from torch.nn import functional as F
from torch.nn.modules import loss
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

class SInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.candidate_num = 14
        vocab_path="pretrained/gpt2-chinese-cluecorpussmall/vocab.txt"
        self.tokenizer = BertTokenizer(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")

    def forward(self, post, input_ids=None, ref=None, labels=None, pseudo=False, opt_idx=0):
        '''
        opt_idx: 0 : extract reference, 1: get generator outputs, 2: get selector outputs 
        '''
        if opt_idx == 0:
            return self.model.extract_M(post, ref)
        else:
            return self.model(post, ref, labels, pseudo)

    def training_step(self, batch, batch_idx):
        post, resp, input_ids, ref = batch["post"], batch["resp"], batch["input_ids"], batch["ref"]
        #train selector
        resp = self.pad_resp(resp, ref)
        ref = torch.cat((resp.unsqueeze(1), ref[:,:-1,:]),dim=1)
        for batch_idx in range(ref.size(0)):
            idx =  torch.randperm(ref[batch_idx, :, :].shape[0])
            label = (idx==0).nonzero()[0]
            cur_ref = ref[batch_idx,idx,:].unsqueeze(0)
            if batch_idx == 0:
                pseudo_label = label
                new_ref = cur_ref
            else:
                pseudo_label = torch.cat((pseudo_label, label), dim=0)
                new_ref = torch.cat((new_ref, cur_ref), dim=0)
        labels = pseudo_label.type_as(ref)
        pseudo_label = self.get_pseudo_label(resp, new_ref)[:,0]
        pseudo_label = (pseudo_label / ref.size(-1)) > 0.2 
        new_ref = new_ref[:,0,:].unsqueeze(1)
        labels = labels.eq(0)
        # if random.random() < 0.7:
        #     # loss,_ = self(post, None, new_ref, labels, False, 2)
        #     loss, _ = self(post, None, new_ref, labels, False, 2)
        # else:
        loss,_ = self(post, None, new_ref, pseudo_label, True, 2) 
        self.log('s_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def get_pseudo_label(self, resp, ref):
        batch_size, max_len = resp.shape
        vocab_size = 21128
        # resp_one_hot = torch.zeros(batch_size*max_len, vocab_size).scatter_(1, resp.view(batch_size*max_len, -1), 1)
        # ref_one_hot = torch.zeros(batch_size*self.candidate_num*max_len, vocab_size).scatter_(1, ref.view(batch_size*self.candidate_num*max_len, -1), 1)
        resp_one_hot = torch.nn.functional.one_hot(resp.view(-1), num_classes=vocab_size)
        ref_one_hot = torch.nn.functional.one_hot(ref.view(-1), num_classes=vocab_size)
        resp_one_hot[:,0] = 0
        ref_one_hot[:,0] = 0 #将pad id置为0
        sub_info = resp_one_hot.view(batch_size, max_len, -1).float()
        ref_one_hot = ref_one_hot.view(batch_size, self.candidate_num, max_len, -1).float()
        select_score = torch.einsum('bclv, bmv -> bclm', ref_one_hot, sub_info)
        select_score = torch.sum(torch.max(select_score, dim=-1)[0], dim=-1)
        # select_score = F.softmax(select_score)
        # pseudo_logits, pseudo_label = select_score.topk(3, dim=1)

        return select_score

    def pad_resp(self, resp, ref):
        ref_len = ref.size(-1)
        resp_len = resp.size(-1)
        if resp_len >= ref_len:
            resp = torch.cat((resp[:, :ref_len-1], resp[:,-1].unsqueeze(1)), dim=-1)
        else:
            resp = torch.cat((resp, torch.zeros((resp.size(0), ref_len-resp_len)).type_as(resp)), dim=-1)
        return resp

    def validation_step(self, batch, batch_idx):
        post, resp, input_ids, ref = batch["post"], batch["resp"], batch["input_ids"], batch["ref"]
        resp = self.pad_resp(resp, ref)
        ref = torch.cat((resp.unsqueeze(1), ref[:,:-1,:]),dim=1)
        label = torch.zeros((ref.size(0))).type_as(ref)
        label = label.eq(0)
        ref = ref[:,0,:].unsqueeze(1)
        loss, logits  = self(post, None, ref, label, False, 2)
        # _, logits = logits.max(-1)
        # logits = logits > 0.5
        # val_acc = sum(logits.eq(label)) / ref.size(0)
        # n_correct, n_word = self.calculate_acc(logits, resp, ignore_index=0)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_acc', n_correct / n_word,
        #          on_step=False, on_epoch=True, prog_bar=True)
        return loss
        # return (n_correct, n_word)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        post, resp, input_ids, ref = batch["post"], batch["resp"], batch["input_ids"], batch["ref"]
        ref_keep = self(post, None, ref, None, False, 0).detach() #[batch_size, turn_num, context_len]
        ref_keep = ref_keep.view(post.size(0), -1)
        post_str = ''.join(self.tokenizer.convert_ids_to_tokens(post.squeeze()))
        ref_str = ''.join(self.tokenizer.convert_ids_to_tokens(ref_keep.squeeze()))
        resp_str =  ''.join(self.tokenizer.convert_ids_to_tokens(resp.squeeze()))
        print('Post: ', post_str)
        print('Ref: ', ref_str)
        print('Resp: ', resp_str)
        return None
        # return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer_d = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return [optimizer_d], []
        else:
            scheduler_d = lrs.StepLR(optimizer_d,
                                    step_size=self.hparams.lr_decay_steps,
                                    gamma=self.hparams.lr_decay_rate)
            scheduler = {"scheduler": scheduler_d, "interval": "step", "frequency": 1}
            return [optimizer_d], [scheduler_d]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        Model = getattr(importlib.import_module(
                '.'+name, package=__package__), name)
        self.model = self.instancialize(Model)
            # self.model.selector.load_weight(self.hparams.pretrained_selector_path)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        # class_args = inspect.getargspec(Model.__init__).args[1:]
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
