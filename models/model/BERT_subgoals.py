from re import L
from transformers import RobertaTokenizer
import numpy as np
import torch.nn as nn
from transformers import AutoModel

import os
import sys
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils2 import sg_idx_to_obj

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

class BERT_Classfication(nn.Module):
    def __init__(self):
        super(BERT_Classfication, self).__init__()
        
        self.base_model = AutoModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.2)
        self.linear_sgtype = nn.Linear(768, 8) # output features from bert is 768 and 2 is number of labels
        self.linear_arg1 = nn.Linear(768, 106)
        self.linear_arg2 = nn.Linear(768, 106)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        outputs = self.dropout(outputs[1])
        outputs_sgtype = self.relu(self.linear_sgtype(outputs))
        outputs_arg1 = self.relu(self.linear_arg1(outputs))
        outputs_arg2 = self.relu(self.linear_arg2(outputs))
        
        return [outputs_sgtype, outputs_arg1, outputs_arg2]

def preprocess(text):
    # Removing punctuations in strings
    punc = '''!()-{};:'"\,./?@#$%^&*_~'''
    # Using loop + punctuation string
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text

def get_target_obj_from_sg_model(sg_model, step_instr):
    x = tokenizer(preprocess(step_instr), padding='max_length', max_length = 60, truncation=True, return_tensors="pt")
    output = sg_model(x['input_ids'].to('cuda'), x['attention_mask'].to('cuda'))
    obj1 = np.argmax(output[1][0][:].cpu().detach().numpy())
    obj2 = np.argmax(output[2][0][:].cpu().detach().numpy())
    if sg_idx_to_obj[obj2] != 'None':
        target_obj_from_sg_model = {'prob': output[2][0][obj2].cpu().detach().numpy(), 'obj': sg_idx_to_obj[obj2]}
    else:
        target_obj_from_sg_model = {'prob': output[1][0][obj1].cpu().detach().numpy(), 'obj': sg_idx_to_obj[obj1]}
    return target_obj_from_sg_model

