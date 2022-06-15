import os
import json
from re import L
import numpy as np
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv

import sys
sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/utils')))
from utils import load_task_json
from utils2 import has_interaction, generate_views, nav_actions

sys.path.append((os.path.join(os.environ['ALFRED_ROOT'], 'models/model')))
from ViltModel import step, get_target_obj_from_vilt
from maskrcnn import target_obj_present
from BERT_subgoals import get_target_obj_from_sg_model

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EvalTask2(Eval):
    '''
    evaluate overall task performance
    '''
    @classmethod
    def run(cls, model, maskrcnn, sg_model, task_queue, args, lock, successes, failures, results):
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
                traj = load_task_json(args, task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, maskrcnn, sg_model, r_idx, traj, args, lock, successes, failures, results)
                cnt += 1

            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, maskrcnn, sg_model, r_idx, traj_data, args, lock, successes, failures, results):
        # reset model
        #model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        step_instr_idx = 0
        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
        # print("GOAL: ", goal_instr)
        # print("STEP INST.: ", step_instr)
        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        
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
                # print('Failed. Max steps reached')
                break

            # generate views
            cur_view, left_view, right_view = generate_views(env)
            t += 4
            # forward model
            m_pred = step(model, maskrcnn, goal_instr, step_instr, action_history, last_collision_action, np.array(cur_view), np.array(left_view), np.array(right_view))
            last_collision_action = None
            
            # check if STOP was predicted
            if m_pred['action_low'] == "Stop":
                consecutive_stops += 1
                if consecutive_stops > args.max_stops:
                    consecutive_stops = 0
                    action_history = []
                    step_instr_idx += 1
                    if step_instr_idx == len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        break
                    else:
                        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                        # print("STEP INST.: ", step_instr)
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
                            # print('Target object detected!')
                            action_history = []
                            step_instr_idx += 1
                            step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                            # print("STEP INST.: ", step_instr)
                            continue
                        else:
                            pred_action = m_pred['stop_case']
                    else:
                        break
                else:
                    consecutive_stops = 0
                    action_history = []
                    step_instr_idx += 1
                    if step_instr_idx == len(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        # print("\tPredicted STOP")
                        break
                    else:
                        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][step_instr_idx]
                        # print("STEP INST.: ", step_instr)
                        continue
            else:
                consecutive_stops = 0
                action_history.append(m_pred['action_low'])
                #get action
                pred_action, pred_mask = m_pred['action_low'], m_pred['action_low_mask']
            
            action_history = action_history[-8:]
                
            # get mask
            pred_mask = pred_mask if has_interaction(pred_action) else None

            # print action
            if args.debug:
                print(pred_action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(pred_action, interact_mask=pred_mask, smooth_nav=args.smooth_nav, debug=args.debug)
            # print(pred_action)
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
                    # print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            # print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward),
                     'failure_reason': failure_reason,
                     'collisions': num_collisions,
                     'first_mani_action_failed': first_action_failed}

        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.log_path)
        save_path = os.path.join(self.args.log_path, 'task_results_v2_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

