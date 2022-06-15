import os
import sys
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/model')))
from BERT_subgoals import BERT_Classfication
from ViltModel import ViltClassifier
from maskrcnn import get_model_instance_segmentation
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils2 import num_actions, num_objects
import torch
import json
import os

def load_model(model_path, maskrcnn_path, sg_model_path):
	device = torch.device("cuda") if torch.cuda.is_available else 'cpu'

	model = ViltClassifier(0.1, num_actions, num_objects)
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()

	no_of_objects = 106
	maskrcnn = get_model_instance_segmentation(no_of_objects)
	maskrcnn.load_state_dict(torch.load(maskrcnn_path))
	maskrcnn.to(device)
	maskrcnn.eval()

	sg_model = BERT_Classfication()
	sg_model.load_state_dict(torch.load(sg_model_path))
	sg_model.to(device)
	sg_model.eval()

	return model, maskrcnn, sg_model
	
def load_task_json(args, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(args.data, task['task'], 'pp/ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data
 
def get_task_root(args, ex):
	'''
	returns the folder path of a trajectory
	'''
	return os.path.join(args.data, ex['split'], *(ex['root'].split('/')[-2:]))
