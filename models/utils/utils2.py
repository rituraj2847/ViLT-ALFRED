import os 
import datetime
import json
import numpy as np
from PIL import Image

idx_to_action = {0: "CloseObject", 1: "SliceObject", 2: "OpenObject", 3: "RotateRight", 4: "Stop", 5: "LookUp", 6: "LookDown", 7: "PutObject", 8: "MoveAhead", 9: "PickupObject", 10: "ToggleObjectOff", 11: "RotateLeft", 12: "ToggleObjectOn"}

idx_to_object = {0: 'SaltShaker', 1: 'Glassbottle', 2: 'WineBottle', 3: 'Pan', 4: 'Box', 5: 'Laptop', 6: 'TennisRacket', 7: 'Egg', 8: 'Apple', 9: 'Candle', 10: 'Spatula',
 11: 'Plate', 12: 'ButterKnife', 13: 'RemoteControl', 14: 'Microwave', 15: 'Safe', 16: 'Cart', 17: 'StoveBurner', 18: 'Drawer', 19: 'BaseballBat', 20: 'CD', 21: 'ToiletPaperHanger',
 22: 'AlarmClock', 23: 'Statue', 24: 'Ladle', 25: 'CounterTop', 26: 'Tomato', 27: 'Fridge', 28: 'Sink', 29: 'SoapBar', 30: 'Bed', 31: 'Potato', 32: 'Plunger', 33: 'FloorLamp',
 34: 'Mug', 35: 'Lettuce', 36: 'Spoon', 37: 'Pen', 38: 'DishSponge', 39: 'Vase', 40: 'Kettle', 41: 'Dresser', 42: 'CellPhone', 43: 'Sofa', 44: 'SprayBottle', 45: 'Newspaper',
 46: 'DeskLamp', 47: 'ArmChair', 48: 'Shelf', 49: 'Cabinet', 50: 'Cup', 51: 'Pot', 52: 'SideTable', 53: 'Watch', 54: 'KeyChain', 55: 'GarbageCan', 56: 'CoffeeMachine',
 57: 'Faucet', 58: 'BasketBall', 59: 'Toilet', 60: 'Bowl', 61: 'HandTowel', 62: 'Desk', 63: 'Bathtub', 64: None, 65: 'CreditCard', 66: 'Book', 67: 'SoapBottle', 68: 'WateringCan',
 69: 'Fork', 70: 'ToiletPaper', 71: 'Bread', 72: 'Pillow', 73: 'Knife', 74: 'Cloth', 75: 'TissueBox', 76: 'Ottoman', 77: 'CoffeeTable', 78: 'Pencil', 79: 'PepperShaker', 80: 'DiningTable'}
 
action_to_idx, object_to_idx = {}, {}
for k in idx_to_action.keys():
    action_to_idx[idx_to_action[k]] = int(k)
for k in idx_to_object.keys():
    object_to_idx[idx_to_object[k]] = int(k)

num_actions = len(action_to_idx)
num_objects = len(object_to_idx)

action_to_lowaction = {"CloseObject": "CloseObject", "SliceObject": "SliceObject", "OpenObject": "OpenObject", "RotateRight": 'RotateRight_90', "Stop": "Stop", "LookUp": "LookUp_15", "LookDown": "LookDown_15", "PutObject": "PutObject", "MoveAhead": 'MoveAhead_25', "PickupObject": "PickupObject", "ToggleObjectOff": "ToggleObjectOff", "RotateLeft": "RotateLeft_90", "ToggleObjectOn": "ToggleObjectOn"}

lowaction_to_action = {}

for k in action_to_lowaction.keys():
    lowaction_to_action[action_to_lowaction[k]] = k


def has_interaction(action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', 'Stop']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True

nav_actions = [act for act in lowaction_to_action.keys() if has_interaction(act) == False]



def save_results2(successes, failures, results, args):
        results = {'successes': list(successes),
                   'failures': list(failures),
                   'results': dict(results)}

        save_path = args.log_path
        #save_path = os.path.join(save_path, 'task_results_' + args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        save_path = os.path.join(save_path, 'task_results_v2_subgoals_' + args.eval_split +  '.json')

        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

def generate_views(env):
    cur_view = Image.fromarray(np.uint8(env.last_event.frame))
    _, event, target_instance_id, err, _ = env.va_interact("RotateLeft_90", interact_mask=None)
    left_view = Image.fromarray(np.uint8(event.frame))
    _, event, target_instance_id, err, _ = env.va_interact("RotateRight_90", interact_mask=None)
    _, event, target_instance_id, err, _ = env.va_interact("RotateRight_90", interact_mask=None)
    right_view = Image.fromarray(np.uint8(event.frame))
    _, event, target_instance_id, err, _ = env.va_interact("RotateLeft_90", interact_mask=None)
    return cur_view, left_view, right_view

"""
for subgoal model
"""

OBJECTS = ['AlarmClock', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cart', 'CellPhone', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'PaintingHanger', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveBurner', 'StoveKnob', 'TVStand', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'Vase', 'Watch', 'WateringCan', 'WineBottle']

i = 1
sg_obj_to_idx = {'None': 0}
sg_idx_to_obj = {0: 'None'}
for obj in OBJECTS:
    sg_obj_to_idx[obj] = i
    sg_idx_to_obj[i] = obj
    i += 1