from re import L
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import numpy as np

STATIC_RECEPTACLES = ['__background__','ArmChair', 'Bathtub', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'HandTowelHolder', 'LaundryHamper', 'Microwave', 'Ottoman', 'PaintingHanger', 'Safe', 'Shelf', 'SideTable', 'Sink', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toaster', 'Toilet', 'ToiletPaperHanger', 'TowelHolder']
OBJECTS = ['__background__', 'AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
ALL = ['__background__', 'AlarmClock', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cart', 'CellPhone', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'PaintingHanger', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveBurner', 'StoveKnob', 'TVStand', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'Vase', 'Watch', 'WateringCan', 'WineBottle']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
    
        
def get_prediction(maskrcnn, img, CLASS_NAMES, confidence):
    img = img.to(device)
    pred = maskrcnn([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if len(pred_t) == 0:
        return None, None, None, None, 1
    pred_t = pred_t[-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [CLASS_NAMES[i+1] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    return masks, pred_boxes, pred_class, pred_score, 0
    
def get_mask(maskrcnn, img, label):
    if label == None:
        return None
    masks, pred_boxes, pred_class, pred_score, flag = get_prediction(maskrcnn, img, ALL, confidence=0.6)
    if flag == 1:
        return None
    for i in range(len(pred_class)):
      if pred_class[i] == label:
        #print("Detected object: ", label)
        mask = masks[i]
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j][k] == True:
                    mask[j][k] = 1
                else:
                    mask[j][k] = 0
        mask = np.asarray(mask)
        # mask = np.array([[1 if mask[j][k] == True else 0 for k in range(mask.shape[1])] for j in range(mask.shape[0])])
        return mask

def target_obj_present(maskrcnn, img, obj):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    masks, pred_boxes, pred_class, pred_score, flag = get_prediction(maskrcnn, img, ALL, confidence=0.4)
    if pred_class == None:
        return False
    for object in pred_class:
        if object == obj:
            return True
    return False
