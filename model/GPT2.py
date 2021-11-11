from transformers import GPT2Config, GPT2LMHeadModel
import torch
import torch.nn as nn
class GPT2(nn.Module):
    def __init__(self, config_path="config/model_config.json") -> None:
        super(GPT2, self).__init__()
        self.model_name = "bert_pretrained_model"
        self.config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.config)
    
    def forward(self, input_ids, ref=None):
        labels = input_ids
        if ref != None:
            input_ids = torch.cat([ref, input_ids], dim=-1)
            labels = torch.cat([torch.ones(ref.shape).type_as(ref)*-100, labels], dim=-1)
        labels = torch.where(labels == 0, -100, labels)
        # attention_mask = attention_mask
        r = self.model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
        )
        return r
    
    def load_weight(self, pretrained_path):
        self.model = self.model.from_pretrained(pretrained_path)


