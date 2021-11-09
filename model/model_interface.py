import importlib
import inspect

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from torch.nn import functional as F
from torch.nn.modules import loss
from transformers import AdamW, get_linear_schedule_with_warmup


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.candidate_num = 14

    def forward(self, post, resp, ref=None, pseudo_label=None, opt_idx = 0):
        '''
        opt_idx: 0 : extract reference, 1: get generator outputs, 2: get selector outputs 
        '''
        if opt_idx == 0:
            return self.model.selector.extract_M(post, ref)
        if opt_idx == 1:
            return self.model.generator(post, resp, ref)
        if opt_idx == 2:
            return self.model.selector(post, ref, pseudo_label)

    # def training_step(self, batch, batch_idx):
    #     inputs, attention_mask = batch["input_ids"], batch["attention_mask"]
    #     outputs = self(inputs, attention_mask)
    #     loss = outputs.loss
    #     self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     return loss
    def training_step(self, batch, batch_idx, optimizer_idx):
        #train generator
        post, resp, ref = batch["post"], batch["resp"], batch["ref"]
        if optimizer_idx == 0:
            ref_keep = self(post, None, ref, None, 0).detach() #[batch_size, turn_num, context_len]
            ref_keep = ref_keep.view(post.size(0), -1)
            outputs = self(post, resp, ref_keep, None, 1)
            loss = outputs.loss
            self.log('g_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        #train selector
        if optimizer_idx == 1:
            outputs = self(post, resp, None, None, 1)
            logits = outputs.logits.detach()
            pseudo_label = self.get_pseudo_label(logits, post, resp, ref)
            loss  = self(post, None, ref, pseudo_label, 2)
            self.log('s_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
    
    def get_pseudo_label(self, logits, post, resp, ref):
        batch_size, max_len, vocab_size = logits.shape
        # resp_one_hot = torch.zeros(batch_size*max_len, vocab_size).scatter_(1, resp.view(batch_size*max_len, -1), 1)
        # ref_one_hot = torch.zeros(batch_size*self.candidate_num*max_len, vocab_size).scatter_(1, ref.view(batch_size*self.candidate_num*max_len, -1), 1)
        resp = torch.cat([post, resp], dim=-1)
        resp_one_hot = torch.nn.functional.one_hot(resp.view(-1), num_classes=vocab_size)
        ref_one_hot = torch.nn.functional.one_hot(ref.view(-1), num_classes=vocab_size)
        sub_info = resp_one_hot.view(batch_size, max_len, -1) - logits
        ref_one_hot = ref_one_hot.view(batch_size, self.candidate_num, max_len // 2, -1).float()
        select_score = torch.einsum('bmv, bclv -> bc', sub_info, ref_one_hot)
        select_score = F.softmax(select_score)
        # pseudo_logits, pseudo_label = select_score.topk(3, dim=1)
        return select_score

    def validation_step(self, batch, batch_idx):
        post, resp, ref = batch["post"], batch["resp"], batch["ref"]
        # print(post, resp, ref)
        ref_keep = self(post, None, ref, None, 0) #[batch_size, turn_num, context_len]
        ref_keep = ref_keep.view(post.size(0), -1)
        outputs = self(post, resp, ref_keep, None, 1)
        loss = outputs.loss
        # logits = outputs.logits
        # n_correct, n_word = self.calculate_acc(logits, resp, ignore_index=0)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_acc', n_correct / n_word,
        #          on_step=False, on_epoch=True, prog_bar=True)
        return loss
        # return (n_correct, n_word)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer_d = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        optimizer_g = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return [optimizer_g, optimizer_d], []
        else:
            scheduler_d = lrs.StepLR(optimizer_d,
                                    step_size=self.hparams.lr_decay_steps,
                                    gamma=self.hparams.lr_decay_rate)
            t_total = 100000
            scheduler_g = get_linear_schedule_with_warmup(optimizer_g, self.hparams.warm_up_steps, t_total)
            scheduler = {"scheduler": scheduler_g, "interval": "step", "frequency": 1}
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
    # def configure_optimizers(self):
    #     if hasattr(self.hparams, 'weight_decay'):
    #         weight_decay = self.hparams.weight_decay
    #     else:
    #         weight_decay = 0
    #     if self.hparams.optimizer == 'Adam':
    #         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
    #     elif self.hparams.optimizer == 'AdamW':
    #         optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
    #     if self.hparams.lr_scheduler is None:
    #         return optimizer
    #     else:
    #         if self.hparams.lr_scheduler == 'step':
    #             scheduler = lrs.StepLR(optimizer,
    #                                    step_size=self.hparams.lr_decay_steps,
    #                                    gamma=self.hparams.lr_decay_rate)
    #         elif self.hparams.lr_scheduler == 'cosine':
    #             scheduler = lrs.CosineAnnealingLR(optimizer,
    #                                               T_max=self.hparams.lr_decay_steps,
    #                                               eta_min=self.hparams.lr_decay_min_lr)
    #         elif self.hparams.lr_scheduler == 'warmup':
    #             t_total = 100000
    #             scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warm_up_steps, t_total)
    #             scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #         else:
    #             raise ValueError('Invalid lr_scheduler type!')
    #         return [optimizer], [scheduler]

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

    def calculate_acc(self, logit, labels, ignore_index=-100):
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = labels[..., 1:].contiguous().view(-1)

        _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
        # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
        non_pad_mask = labels.ne(ignore_index)
        n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return n_correct, n_word

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocab size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
            # ...表示其他维度由计算机自行推断
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        # camel_name = ''.join([i.capitalize() for i in name.split('_')])
        # try:
        #     Model = getattr(importlib.import_module(
        #         '.'+name, package=__package__), camel_name)
        # except:
        #     raise ValueError(
        #         f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        Model = getattr(importlib.import_module(
                '.'+name, package=__package__), name)
        self.model = self.instancialize(Model)
        if self.hparams.pretrained:
            self.model.generator.load_weight(self.hparams.pretrained_generator_path)
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
