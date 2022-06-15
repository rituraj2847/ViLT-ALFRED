import os
import sys
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import torch
import json
import argparse
import numpy as np
from datetime import datetime

# from eval_task import EvalTask
from env.thor_env import ThorEnv
import torch.multiprocessing as mp

sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils import load_task_json
from utils2 import has_interaction, generate_views, nav_actions

sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/model')))
from ViltModel import step, get_target_obj_from_vilt
from maskrcnn import target_obj_present
from BERT_subgoals import get_target_obj_from_sg_model

import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# class Leaderboard(EvalTask):
class Leaderboard(Eval):
    '''
    dump action-sequences for leaderboard eval
    '''
    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        # load model
        print("Loading model: ", self.args.model_path)
        print("Loading maskrcnn: ", self.args.maskrcnn_path)
        print("Loading subgoal model: ", self.args.sg_model_path)
        self.model, self.maskrcnn, self.sg_model = load_model(self.args.model_path, self.args.maskrcnn_path, self.args.sg_model_path)
        self.model.share_memory()
        self.maskrcnn.share_memory()
        self.sg_model.share_memory()
        # # gpu
        # if self.args.gpu:
        #     self.model = self.model.to(device)

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))
        
    @classmethod
    def run(cls, model, maskrcnn, sg_model, task_queue, args, lock, splits, seen_actseqs, unseen_actseqs):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                # print("Task: ",task, task["task"])
                traj = load_task_json(args, task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env,  model, maskrcnn, sg_model, r_idx, traj, args, lock, splits, seen_actseqs, unseen_actseqs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, maskrcnn, sg_model, r_idx, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):
        # reset model
        # model.reset()

        # setup scene
        cls.setup_scene(env, traj_data, r_idx, args)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        step_instr_idx = 0
        print("Total Step instructions to take place: ", len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']))
        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
        print("GOAL: ", goal_instr)
        print("STEP INST.: ", step_instr)

        done, success = False, False
        actions = list()
        fails = 0
        t = 0
        action_history = []
        last_collision_action = None
        failure_reason = None
        first_action_failed = None
        num_collisions = 0
        consecutive_stops = 0

        while not done:
        
            # break if max_steps reached
            if t >= args.max_steps:
                failure_reason = 'max steps'
                break

            # generate views
            cur_view, left_view, right_view = generate_views(env)
            t += 4
            # forward model
            m_pred = step(model, maskrcnn, goal_instr, step_instr, action_history, last_collision_action, np.array(cur_view), np.array(left_view), np.array(right_view))
            last_collision_action = None
            # check if STOP was predicted
            if m_pred['action_low'] == "Stop":
                # print("Predicted STOP")
                consecutive_stops += 1
                if consecutive_stops > args.max_stops:
                    consecutive_stops = 0
                    action_history = []
                    step_instr_idx += 1
                    if step_instr_idx == len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        # print("\tPredicted STOP with consecutive steps > Max Steps")
                        break
                    else:
                        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                        # print("Continuing to Next STEP INST. with consecutive steps > Max Steps (defined by ourself): ", step_instr)
                        continue
                if len(action_history) != 0 and action_history[-1] in nav_actions:
                    if step_instr_idx + 1 != len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        next_step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx+1]
                        target_obj_from_vilt = get_target_obj_from_vilt(model, maskrcnn, goal_instr, next_step_instr, [], np.array(cur_view), np.array(left_view), np.array(right_view))
                        target_obj_from_sg_model = get_target_obj_from_sg_model(sg_model, next_step_instr)
                        # print("Target obj from vilt", target_obj_from_vilt['obj'])
                        # print('Target object from sg model', target_obj_from_sg_model['obj'])

                        if target_obj_from_vilt['prob'] > target_obj_from_sg_model['prob']:
                            target_obj = target_obj_from_vilt['obj']
                        else:
                            target_obj = target_obj_from_sg_model['obj']

                        # print('Target obj chosen: ', target_obj)
                        if target_obj == None or target_obj_present(maskrcnn, cur_view, target_obj):
                            print('Target object detected!')
                            action_history = []
                            step_instr_idx += 1
                            step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                            # print("Continuing to Next STEP INST.: ", step_instr)
                            continue
                        else:
                            # print('Target object not detected!, trying to explore')
                            pred_action = m_pred['stop_case']
                    else:
                        # print("\tPredicted STOP with step_instr_idx as last instruction")
                        break
                else:
                    consecutive_stops = 0
                    action_history = []
                    step_instr_idx += 1
                    if step_instr_idx == len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        # print("\tPredicted STOP with this being the last instr")
                        break
                    else:
                        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                        # print("Next STEP INST. as stop was predicted and might be mani action: ", step_instr)
                        continue
            else:
                consecutive_stops = 0
                action_history.append(m_pred['action_low'])
                #get action
                pred_action, pred_mask = m_pred['action_low'], m_pred['action_low_mask']
            
            action_history = action_history[-8:]
                
            # get mask
            pred_mask = pred_mask if has_interaction(pred_action) else None


            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, api_action = env.va_interact(pred_action, interact_mask=pred_mask, smooth_nav=False)
            # print("API Action being sent: ",api_action)
            # print("Pred Action: ", pred_action)

            if not t_success:
                # print("Action failed!")
                last_collision_action = pred_action
                if last_collision_action == "MoveAhead_25":
                    num_collisions += 1 
                if first_action_failed is None and has_interaction(last_collision_action) is True:
                    first_action_failed = last_collision_action
                fails += 1
                if fails >= args.max_fails:
                    failure_reason = 'max fails'
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
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

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']


        # print("--------------------Limiting File Size for sanity checks-----------")
        # seen_files = seen_files[5:7]
        # unseen_files = unseen_files[5:6]

        # add seen trajectories to queue
        for traj in seen_files:
            task_queue.put(traj)

        # add unseen trajectories to queue
        for traj in unseen_files:
            task_queue.put(traj)
        # print("Inside Queue_task: ",task_queue)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        self.model.test_mode = True
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.maskrcnn, self.sg_model, task_queue, self.args, lock,
                                                       self.splits, self.seen_actseqs, self.unseen_actseqs))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = args.log_path
        # save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_feat_2.1.0")
    parser.add_argument('--model_path', type=str, default="/path/to/June5_model_optim_scheduler_3_300.pt")
    parser.add_argument('--maskrcnn_path', type=str, default='/path/to/mrcnn_alfred_all_009.pth')
    parser.add_argument('--sg_model_path', type=str, default='/path/to/RoBERTa_base_subgoals.pth')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')
    parser.add_argument('--max_stops', type=int, default=5, help='max stops')
    parser.add_argument('--log_path', type=str, default='/path/to/logs')


    #eval_split argument not to be used
    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()
