from turtle import right
from transformers import ViltModel, ViltFeatureExtractor, BertTokenizerFast
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as T
import random

import os
import sys
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils2 import *

sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/model')))
from maskrcnn import get_mask

"""
Model definition
"""
class ViltClassifier(nn.Module):
    def __init__(self, p, num_actions, num_objects):
        super(ViltClassifier, self).__init__()
        
        self.base = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.dropout = nn.Dropout(p)
        self.linear = nn.Linear(768*2, 2048)
        # self.linear = nn.Linear(768, 2048)        
        self.linear1 = nn.Linear(2048, num_actions)
        self.linear2 = nn.Linear(2048, num_objects)
        self.linear3 = nn.Linear(2048, num_objects)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.base(**x)
        output = torch.cat((self.relu(self.dropout(output['pooler_output'])), self.relu(self.dropout(output['last_hidden_state'][:, 40, :]))), dim=1)
        output = self.relu(self.linear(output))

        action_type = self.linear1(output)
        obj1 = self.linear2(output)
        obj2 = self.linear3(output)
        
        return [action_type, obj1, obj2]
        
"""
Dataset creation and processing
""" 
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.remove('on')
stop_words.remove('above')
stop_words.remove('below')
stop_words.remove('in')
stop_words.remove('under')

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
ft = ViltFeatureExtractor(size=420)

def preprocess(goal, step_inst, action_hist):
    inp = goal + " STEP " +  step_inst
    inp = ' '.join([word.lower() for word in inp.split() if word.lower() not in stop_words])
    inp = inp .replace('.', '')
    inp = inp .replace(',', '')
    inp = [inp, ' '.join([str(action_to_idx[action]) for action in action_hist])]
    return inp 
    
    
def step(model, maskrcnn, goal, step_instr, action_history, last_collision_action, front_view, left_view, right_view):
    #print("Goal: ", goal, "\nStep Ins: ", step_instr, "\nAction History: ", action_history)

    action_hist = []
    for i in range(len(action_history)):
        action_hist.append(lowaction_to_action[action_history[i]])
    
    with torch.no_grad():
        img = np.concatenate([front_view, left_view, right_view], axis=0)
        img_f = ft(img, return_tensors='pt')

        processed_text = preprocess(goal, step_instr, action_hist)
        text_f = tokenizer([processed_text], max_length=40, padding='max_length', truncation=True, return_tensors='pt')
        X = dict(text_f)
        X.update(img_f)
        for k in X.keys():
            X[k] = X[k].to(device)

        output = model(X)
        
        up_idx = action_to_idx['LookUp']
        down_idx = action_to_idx['LookDown']
        left_idx = action_to_idx['RotateLeft']
        right_idx = action_to_idx['RotateRight']
        ahead_idx = action_to_idx['MoveAhead']

        # In case of Stop prediction, and Goto subgoal, choosing one of the Nav actions with highest probability
        nav_actions = ['LookUp', 'LookDown', 'RotateLeft', 'RotateRight', 'MoveAhead']
        nav_actions_probs = {}
        for action, idx in zip(nav_actions, [up_idx, down_idx, left_idx, right_idx, ahead_idx]):
            nav_actions_probs[action] = output[0][0][idx].cpu().detach().numpy()

        nav_actions_probs = dict(sorted(nav_actions_probs.items(), key=lambda item: item[1]))

        # In case of collision, choose one of the RotateLeft, RotateRight
        if last_collision_action != None and 'MoveAhead' == lowaction_to_action[last_collision_action]: 
            if nav_actions_probs['RotateLeft'] > nav_actions_probs['RotateRight']:
                return {'action_low': action_to_lowaction['RotateLeft'], 'action_low_mask': None}
            else:
                return {'action_low': action_to_lowaction['RotateRight'], 'action_low_mask': None}
        else:
            pred_action = [idx_to_action[np.argmax(output[0][0][:].cpu().detach().numpy())], \
                        idx_to_object[np.argmax(output[1][0][:].cpu().detach().numpy())], \
                        idx_to_object[np.argmax(output[2][0][:].cpu().detach().numpy())]]


        mask_obj = pred_action[1]
        transform = T.Compose([T.ToTensor()])
        img = transform(front_view)        
        if pred_action[2] != None:
            mask_obj = pred_action[2]
        obj1 = np.argmax(output[1][0][:].cpu().detach().numpy())
        obj2 = np.argmax(output[2][0][:].cpu().detach().numpy())
        if idx_to_object[obj2] != None:
            prob = output[2][0][obj2].cpu().detach().numpy()
        else:
            prob = output[1][0][obj1].cpu().detach().numpy()
        m_pred = {'action_low': action_to_lowaction[pred_action[0]], 'target_obj': pred_action[1], 'receptacle': pred_action[2], 'prob': prob, 'stop_case': action_to_lowaction[list(nav_actions_probs.keys())[-1*random.randint(0,2)]], 'action_low_mask': get_mask(maskrcnn, img, mask_obj)}
        return m_pred
     
def get_target_obj_from_vilt(model, maskrcnn, goal, step_instr, action_history, front_view, left_view, right_view):
    m_pred = step(model, maskrcnn, goal, step_instr, action_history, None, front_view, left_view, right_view)

    if m_pred['receptacle'] != None:
        target_obj_from_vilt = {'prob': m_pred['prob'], 'obj': m_pred['receptacle']}
    else:
        target_obj_from_vilt = {'prob': m_pred['prob'], 'obj': m_pred['target_obj']}

    return target_obj_from_vilt
    
