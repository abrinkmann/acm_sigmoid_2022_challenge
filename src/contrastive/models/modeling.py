import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from transformers import AutoModel, AutoConfig
from src.contrastive.models.loss import SupConLoss

from pdb import set_trace

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseEncoder(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        
        output = self.transformer(input_ids, attention_mask)

        return output

class ContrastivePretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj=32, temperature=0.07):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        self.transform = nn.Linear(self.config.hidden_size, self.proj)

        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = torch.tanh(self.transform(output))

        output = F.normalize(output, dim=-1)

        loss = self.criterion(output, labels)

        return ((loss,))