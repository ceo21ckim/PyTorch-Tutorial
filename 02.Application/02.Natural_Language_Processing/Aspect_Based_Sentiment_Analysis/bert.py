# coding=utf-8
# Copyright 2018 Google AI Language, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertPooler
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

import torch 
from transformers import BertModel 
from seq_utils import * 
from torch import nn 

from bert_utils import *



BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
 'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin',
 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin',
 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin',
 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin',
 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin',
 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin',
 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin',
 'bert-base-german-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin',
 'bert-large-uncased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin',
 'bert-large-cased-whole-word-masking': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin',
 'bert-large-uncased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin',
 'bert-large-cased-whole-word-masking-finetuned-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin',
 'bert-base-cased-finetuned-mrpc': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin',
 'bert-base-german-dbmdz-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin',
 'bert-base-german-dbmdz-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin'
}

XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {
 'xlnet-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin',
 'xlnet-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin'
}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                  
###########################################################################################################################
            
class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768

class SAN(nn.Module):
    def __init__(self, model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.model = model 
        self.self_attn = nn.MultiheadAttention(model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(model)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask) # (key, query, value)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src 

    
class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.tagger_config = TaggerConfig()
        self.bert = BertModel(bert_config)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if bert_config.fix_tfm:
            for p in self.bert.parameters():
                p.required_grad = False  # Frizen
        
        self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
        self.tagger = SAN(model=bert_config.hidden_size, nhead=12, dropout=0.1)
        penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, self.num_labels)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, 
            head_mask=head_mask
        )
        
        tagger_input = outputs[0] # pooler 
        tagger_input = self.bert_dropout(tagger_input)
        tagger_input = tagger_input.transpose(0, 1) # 각 성분에 대해서 classification을 하기 위함.
        classifier_input = self.tagger(tagger_input)
        classifier_input = classifier_input.transpose(0, 1)
        classifier_input = self.tagger_dropout(classifier_input)
        logits = self.classifier(classifier_input)
        
        outputs = (logits, ) + outputs[2:]
        
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = criterion(active_logits, active_labels)
            
            else:
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss, ) + outputs 
        
        return outputs 