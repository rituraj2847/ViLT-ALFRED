import numpy as np

import os
import sys
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/model')))
import ViltModel

sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils2 import generate_views, has_interaction
from utils import load_model

from env.thor_env import ThorEnv
import json

def setup_scene(env, traj_data, r_idx):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

def replay(env, traj_data, r_idx, model, maskrcnn):
    setup_scene(env, traj_data, r_idx)

    goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    step_instr_idx = 0
    step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
    print("GOAL: ", goal_instr)
    print("AT STEP INST.: ", step_instr)
    done = False

    action_history = []
    last_collision_action = None
    image_id = 0
    while not done:

        # generate views
        cur_view, left_view, right_view = generate_views(env)
        print("Current View Type", type(cur_view))
        # im = Image.fromarray(cur_view)
        cur_view.save("/media/t2r/HDD/final/logs/{}_{}.jpeg".format(goal_instr, image_id))
        image_id += 1

        # forward model
        m_pred = ViltModel.step(model, maskrcnn, goal_instr, step_instr, action_history, last_collision_action, np.array(cur_view), np.array(left_view), np.array(right_view))
        last_collision_action = None
        
        # check if STOP was predicted
        if m_pred['action_low'] == "Stop":
            action_history = []
            step_instr_idx += 1
            if step_instr_idx == len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                print("\tPredicted STOP")
                break
            else:
                step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                print("AT STEP INST: ", step_instr)
            continue
        else:                     
            action_history.append(m_pred['action_low'])
        
        action_history = action_history[-8:]
            
        # get action and mask
        pred_action, pred_mask = m_pred['action_low'], m_pred['action_low_mask']
        pred_mask = pred_mask if has_interaction(pred_action) else None

        # print action
        print(pred_action)

        # use predicted action and mask (if available) to interact with the env
        t_success, _, _, err, _ = env.va_interact(pred_action, interact_mask=pred_mask)

    print("Goal Reached")

if __name__ == '__main__':
    model_path = "/media/t2r/HDD/final/model_save/June5_model_optim_scheduler_3_300.pt"
    maskrcnn_path = '/media/t2r/HDD/final/model_save/mrcnn_alfred_all_009.pth'
    sg_model_path = '/media/t2r/HDD/final/model_save/RoBERTa_base_subgoals.pth'

    model, maskrcnn, _ = load_model(model_path, maskrcnn_path, sg_model_path)

    env = ThorEnv()
    json_path = '/home/t2r/alfred/data/json_feat_2.1.0/pick_heat_then_place_in_recep-Egg-None-CounterTop-2/trial_T20190908_122951_021026/pp/ann_0.json'
    # json_path = '/home/t2r/alfred/data/json_feat_2.1.0/pick_two_obj_and_place-SoapBar-None-GarbageCan-418/trial_T20190909_055649_717880/pp/ann_0.json'
    # json_path = '/home/t2r/alfred/data/json_feat_2.1.0/pick_cool_then_place_in_recep-Tomato-None-Microwave-13/trial_T20190910_173916_331859/pp/ann_1.json'
    # json_path = '/home/t2r/alfred/data/json_feat_2.1.0/look_at_obj_in_light-KeyChain-None-DeskLamp-327/trial_T20190906_202601_548273/pp/ann_2.json'
    with open(json_path) as f:
        data = json.load(f)

    replay(env, data, 0, model, maskrcnn)

