# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from cgi import test
import os
import torch
import cv2
import random
import numpy as np
import pdb
import copy
import argparse
import json
import glob
from tqdm import tqdm
from types import SimpleNamespace

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from utils.hands23.hodetector.data import register_ho_pascal_voc, hoMapper
from utils.hands23.hodetector.modeling import roi_heads
from utils.hands23.vis_utils import vis_per_image


def parse_grasp(grasp_type):
    # get names for grasp
    if grasp_type == 0:
        return "NP-Palm"
    elif grasp_type == 1:
        return  "NP-Fin"
    elif grasp_type == 2:
        return "Pow-Pris"
    elif grasp_type == 3:
        return "Pre-Pris"
    elif grasp_type == 4:
        return "Pow-Circ"
    elif grasp_type == 5:
        return "Pre-Circ"
    elif grasp_type == 6:
        return  "Later"
    elif grasp_type == 7:
        return "Other"
    else:
        print("Can not parse grasp: {grasp_type}")
        # pdb.set_trace()

def parse_contact(contact):
    # get names for contact
    if contact == 0:
        return "no_contact"
    elif contact == 1:
        return "other_person_contact"
    elif contact == 2:
        return "self_contact"
    elif contact == 3:
        return "object_contact"
    elif contact == 4:
        return "obj_to_obj_contact"  
    else:
        print("Can not parse contact: {contact}")
        # pdb.set_trace()
                
def parse_touch(touch):
    # get names for touch
    if touch == 0:
        return "tool_,_touched"
    elif touch == 1:
        return "tool_,_held"
    elif touch == 2:
        return "tool_,_used"
    elif touch == 3:
        return "container_,_touched"
    elif touch == 4:
        return "container_,_held"
    elif touch == 5:
        return "neither_,_touched"
    elif touch == 6:
        return "neither_,_held"
    else:
        print("Can not parse touch: {touch}")
        return None
        # pdb.set_trace()



class Hands:
    def __init__(self, hand_id, hand_bbox, hand_mask, contactState, hand_side, grasp, pred_score, grasp_scores = None):
        self.id = hand_id
        self.hand_bbox = hand_bbox
        self.contactState = parse_contact(contactState)
        self.hand_side = "right" if hand_side==1 else "left"
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_score = None
        self.second_obj_bbox = None
        self.grasp = parse_grasp(grasp)
        self.grasp_scores = grasp_scores
        self.hand_mask = hand_mask
        self.pred_score = round(pred_score,2)
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_clean = None
        self.obj_masks = None
        self.second_obj_bbox = None
        self.second_obj_masks = None
        self.has_first = False
        self.has_second = False
        self.obj_pred_score = None
        self.sec_obj_pred_score = None
    

    def set_first_obj(self, obj_bbox , obj_touch , obj_masks, pred_score, touch_scores = None):
        self.obj_bbox = obj_bbox
        self.obj_touch = parse_touch(obj_touch)
        self.obj_masks = obj_masks
        self.obj_pred_score = round(pred_score,2)
        self.has_first  = True
        self.obj_touch_score = touch_scores
         

    def set_second_obj(self, obj_bbox, obj_masks, pred_score):
        self.second_obj_bbox = obj_bbox
        self.second_obj_masks = obj_masks
        self.sec_obj_pred_score = round(pred_score,2)
        self.has_second = True
    

    def save_masks(self, save_dir, im, img_id, mess = ''):
        ims = copy.deepcopy(im)
        ims[:,:,:] = 0
        ims[self.hand_mask, :] = 255

      
        save_dir = os.path.join(save_dir, "masks"+mess)
        img_id = img_id.strip('\n')
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(save_dir+'/2_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

        if self.has_first:
            ims[:,:,:] = 0
            ims[self.obj_masks, :] = 255
            cv2.imwrite(save_dir+'/3_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

            if self.has_second:
                ims[:,:,:] = 0
                ims[self.second_obj_masks, :] = 255
                cv2.imwrite(save_dir+'/5_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

    def parse_grasp_scores(self):
        grasp_dict = {}
        for type, score in zip(["NP-Palm","NP-Fin", "Pow-Pris", "Pre-Pris", "Pow-Circ", "Pre-Circ", "Later","Other"], self.grasp_scores):
            grasp_dict[type] = str(round(score.item(),4))
        return grasp_dict


    def parse_touch_scores(self):
        touch_dict = {}
        for touch,score in zip(["tool_,_touched", "tool_,_held", "tool_,_used", "container_,_touched", "container_,_held","neither_,_touched", "neither_,_held"], self.obj_touch_score):
            touch_dict[touch] = str(round(score.item(),4))
        return touch_dict

    def get_json(self):
        info = {}
        info['hand_id'] = self.id 
        info['hand_bbox'] = [str(x) for x in self.hand_bbox]
        info['contact_state'] = self.contactState
        info['hand_side'] = self.hand_side
        info['obj_bbox'] = [str(x) for x in self.obj_bbox] if self.obj_bbox is not None else None
        info['obj_touch'] = str(self.obj_touch)
        info['obj_touch_scores'] = self.parse_touch_scores() if self.has_first else None
        info['second_obj_bbox'] = [str(x) for x in self.second_obj_bbox]  if self.second_obj_bbox is not None else None
        info['grasp'] = self.grasp
        info['grasp_scores'] = self.parse_grasp_scores() 
        info['hand_pred_score'] = str(self.pred_score)
        info['obj_pred_score'] = str(self.obj_pred_score)
        info['sec_obj_pred_score'] = str(self.sec_obj_pred_score)
        return info
    
    def message(self):
        return str(self.hand_side) + ' | ' + str(self.contactState) + ' | ' + str(self.hand_bbox).replace('[','').replace(']','') + ' | ' +  str(self.obj_bbox).replace('[','').replace(']','') + ' | ' +str(self.obj_touch) + ' | ' + str(self.second_obj_bbox).replace('[','').replace(']','') + " | " + str(self.grasp) +'\n'

    def get_hands_data_for_memory(self):
        return {
            "which_hand": self.hand_side,
            "contactWhat": self.contactState,
            "touchState": self.obj_touch
        }
        

cropped_object_count = 0

# TODO: purpose is to crop the object
def crop_and_save_object(image, obj_bbox, save_dir, obj_id):
    """
    Crop and save the object from the given bounding box.

    Parameters:
    - image: The original image (numpy array)
    - obj_bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
    - save_dir: Directory to save the cropped object image
    - obj_id: Unique identifier for the object
    """
    x_min, y_min, x_max, y_max = map(int, obj_bbox)
    cropped_object = image[y_min:y_max, x_min:x_max]
    print("------ENTERED-------")
    if cropped_object.size == 0:
        print(f"Warning: Empty cropped object for obj_id: {obj_id}")
        return
    global cropped_object_count
    cropped_object_path = os.path.join(save_dir, f'cropped_object_{cropped_object_count}.png')
    cropped_object_count += 1
    cv2.imwrite(cropped_object_path, cropped_object)
    print(f"Saved cropped object to {cropped_object_path}")
    return cropped_object_path # TODO: made changes to this


# TODO: purpose of this function is to avoid duplicates
# Previously, if there are 2 hands holding the same objects, the same object will be cropped twice
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max]
    
    Returns:
    - iou: IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute the area of intersection
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Compute the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    
    # print("box1_area", box1_area)
    # print("box2_area", box2_area)
    
    # Compute the IoU
    # iou = inter_area / (box1_area + box2_area - inter_area)
    
    iou = inter_area / min(box1_area, box2_area)
    
    smaller_index = 0 if box1_area < box2_area else 1
    # print("smaller_index", smaller_index)

    return iou, smaller_index


def deal_output(im, predictor, save_dir):
    outputs = predictor(im)

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
    pred_classes = outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()

    interaction = torch.tensor(pred_dz[:, 4])
    hand_side = torch.tensor(pred_dz[:, 5])
    grasp = torch.tensor(pred_dz[:, 6])
    touch_type = torch.tensor(pred_dz[:, 7])
    contact_state = torch.tensor(pred_dz[:,8])

    grasp_scores = torch.tensor(pred_dz[:,10:18])
    touch_scores = torch.tensor(pred_dz[:,18:25])

    hand_list = []
    processed_objects = []  # To store processed objects based on IoU
    count = 0

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:
            curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], 
                             contactState=int(contact_state[i].item()), hand_side=hand_side[i].item(), 
                             grasp = grasp[i].item(), pred_score= pred_scores[i], grasp_scores= grasp_scores[i])
            count += 1

            if interaction[i] >= 0:
                obj_id = int(interaction[i])
                obj_bbox = pred_boxes[obj_id]

                # Check if the object is already processed using IoU threshold
                is_duplicate = False
                for processed_bbox in processed_objects:
                    if calculate_iou(obj_bbox, processed_bbox) > 0.7:  # IoU threshold
                        is_duplicate = True
                        break

                if not is_duplicate:
                    curr_hand.set_first_obj(obj_bbox=obj_bbox, obj_touch=touch_type[obj_id].item(), 
                                           obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], 
                                           touch_scores=touch_scores[obj_id])
                    crop_and_save_object(im, obj_bbox, save_dir, obj_id)
                    processed_objects.append(obj_bbox)

                if interaction[obj_id] >= 0:
                    second_obj_id = int(interaction[obj_id])
                    second_obj_bbox = pred_boxes[second_obj_id]

                    is_duplicate = False
                    for processed_bbox in processed_objects:
                        if calculate_iou(second_obj_bbox, processed_bbox) > 0.7:  # IoU threshold
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        curr_hand.set_second_obj(obj_bbox=second_obj_bbox, obj_masks=pred_masks[second_obj_id], 
                                                 pred_score= pred_scores[second_obj_id])
                        crop_and_save_object(im, second_obj_bbox, save_dir, second_obj_id)
                        processed_objects.append(second_obj_bbox)

            hand_list.append(curr_hand)

    return hand_list


def set_cfg(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = args.model_weights

    # assign values to the thresholds for bounding box prediction
    cfg.HAND = args.hand_thresh
    cfg.FIRSTOBJ = args.first_obj_thresh
    cfg.SECONDOBJ = args.second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(args.hand_thresh, args.first_obj_thresh, args.second_obj_thresh)

    # assign values to the thresholds for interaction classification
    cfg.HAND_RELA = args.hand_rela
    cfg.OBJ_RELA = args.obj_rela

    # cfg.MODEL.DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg.freeze()

    return cfg

def hands_detector():
    # TODO: to move this into utils, rename main and hardcode all the values in 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hand_thresh", type=float, default=0.99) # TODO: changed the threshold to 0.99 to avoid irrelevant/small hands
    # parser.add_argument("--first_obj_thresh", type=float, default=0.5)
    # parser.add_argument("--second_obj_thresh", type=float, default=0.3)
    # parser.add_argument("--hand_rela", type=float, default=0.3)
    # parser.add_argument("--obj_rela", type=float, default=0.7)
  
    # parser.add_argument("--model_weights", default=f"./models/model_hands23.pth")
    # parser.add_argument("--data_dir", default = 'demo_example/example_images/')
    # parser.add_argument("--save_dir", default="results/demo")
    # parser.add_argument("--save_img", type=bool,  default = True)
    # parser.add_argument("--image_list_txt", default=None)
    # parser.add_argument("--config_file", default="./config/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
    # args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args_old = {
        "hand_thresh": 0.99,  
        "first_obj_thresh": 0.5,
        "second_obj_thresh": 0.3,
        "hand_rela": 0.3,
        "obj_rela": 0.7,
        "model_weights": os.path.join(script_dir, "../../models/model_hands23.pth"),
        "data_dir": os.path.join(script_dir, "../../my_images/"),
        "save_dir": os.path.join(script_dir, "../../results/my_images/"),
        "save_img": True,
        "image_list_txt": None,
        "config_file": os.path.join(script_dir, "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
    }

    args = SimpleNamespace(**args_old)
    # set configuration
    cfg = set_cfg(args)
    
    predictor = DefaultPredictor(cfg)
    
    # inputs
    if args.image_list_txt is not None:
        f = open(args.image_list_txt)
        images = f.readlines()
    else:
        images = [x.replace(args.data_dir, '') for x in glob.glob(f'{args.data_dir}/*')]

    # outputs
    save_dir = args.save_dir
    save_mask_dir = f"{save_dir}/masks" 
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    save_img = args.save_img 

    # save results
    res = {}
    res["save_dir"] = save_dir
    res["images"] = []
    json_path = os.path.join(save_dir, "result.json")

    # loop
    for test_img in tqdm(images):
        test_img = test_img.strip("\n")
        print(f'Processing: {test_img}')
        # if os.path.exists(os.path.join(args.data_dir, test_img)) is False:
        #     print(os.path.join(args.data_dir, test_img))
        #     pdb.set_trace()
        #     continue
        im = cv2.imread(os.path.join(args.data_dir, test_img))
        im_name = os.path.split(test_img)[-1][:-4]
        
        # record img res
        img = {}
        img["file_name"] = test_img
        img["predictions"] = []

        #save masks and vis
        print("Im: {}, image_name: {}", im, )
        hand_lists = deal_output(im = im, predictor = predictor, save_dir = args.save_dir)

        for hands in hand_lists:

            if save_img:
                hands.save_masks(save_dir, im, test_img.split('/')[-1])

            img['predictions'].append(hands.get_json())
        
        # vis and save TODO: remove if we don't want the additional image
        if save_img:
            im = vis_per_image(im, img['predictions'], im_name+'.png', save_mask_dir, use_simple=False)
            save_path = os.path.join(save_dir, im_name+'.png')
            im.save(save_path)

        res["images"].append(img)

    f = open(json_path, 'w')
    json.dump(res, f, indent=4)

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
def draw_and_save_bounding_boxes(image, hand_interaction_info, save_path, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image based on hand_interaction_info and save it.

    Parameters:
    - image: The original image (numpy array)
    - hand_interaction_info: Dictionary containing hand interaction information
    - save_path: Path to save the image with the bounding boxes
    - color: Color of the bounding boxes (default is green)
    - thickness: Thickness of the bounding box lines (default is 2)
    """
    image_with_boxes = image.copy()

    for hand_side, info in hand_interaction_info.items():
        if hand_side == "left" or hand_side == "right":
            # Ensure the bounding box coordinates are integers
            if info['obj_bbox'] is not None:
                x_min, y_min, x_max, y_max = map(int, info['obj_bbox'])

                # Draw the bounding box on the image
                cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, thickness)


    # cv2.imwrite(save_path, image_with_boxes)
    # print(f"Image with bounding boxes saved to {save_path}")

def crop_image_with_bbox(image, bbox):
    """
    Crop the image using the provided bounding box.

    Parameters:
    - image: The original image (numpy array)
    - bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)

    Returns:
    - Cropped image
    """
    # Ensure the bounding box coordinates are integers
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Crop the image using the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


class HandsDetector:
    def __init__(self):

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use MPS on Apple Silicon
        else:
            self.device = torch.device("cuda")  # Default to CPU
        
        print(f"Using device: {self.device}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        args_old = {
            "hand_thresh"       : 0.80,  
            "first_obj_thresh"  : 0.5,
            "second_obj_thresh" : 0.3,
            "hand_rela"         : 0.3,
            "obj_rela"          : 0.7,
            "model_weights"     : os.path.join(script_dir, "../../models/model_hands23.pth"),
            "data_dir"          : os.path.join(script_dir, "../../my_images/"),
            "save_dir"          : os.path.join(script_dir, "../../results/my_images/"),
            "save_img"          : True,
            "image_list_txt"    : None,
            "config_file"       : os.path.join(script_dir, "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
        }
        # set configuration and predictor
        args = SimpleNamespace(**args_old)
        cfg = set_cfg(args)
        cfg.defrost()
        cfg.MODEL.DEVICE = str(self.device)  # Convert to string

        self.predictor = DefaultPredictor(cfg)
        # self.predictor.model.to(self.device)
        
    def detect_hands(self, image):
        # cv2.imwrite("detect_hands.png", image)
        outputs = self.predictor(image)

        pred_boxes   = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
        pred_dz      = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
        pred_classes = outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
        pred_scores  = outputs["instances"].get("scores").to("cpu").detach().numpy()
        pred_masks   = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()

        interaction   = torch.tensor(pred_dz[:, 4])
        hand_side     = torch.tensor(pred_dz[:, 5])
        grasp         = torch.tensor(pred_dz[:, 6])
        touch_type    = torch.tensor(pred_dz[:, 7])
        contact_state = torch.tensor(pred_dz[:, 8])

        grasp_scores = torch.tensor(pred_dz[:,10:18])
        touch_scores = torch.tensor(pred_dz[:,18:25])

        processed_objects = []
        hand_list = []
        count = 0

        # print("len(pred_classes), ", len(pred_classes))
                
        for i in range(len(pred_classes)):
            if pred_classes[i] == 0: # if the class is hand
                curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], 
                                contactState=int(contact_state[i].item()), hand_side=hand_side[i].item(), 
                                grasp = grasp[i].item(), pred_score= pred_scores[i], grasp_scores= grasp_scores[i])
                count += 1
                
                if interaction[i] >= 0:
                    obj_id = int(interaction[i])
                    obj_bbox = pred_boxes[obj_id]

                    # Check if the object is already processed using IoU threshold
                    curr_hand.set_first_obj(obj_bbox=obj_bbox, obj_touch=touch_type[obj_id].item(), 
                                        obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], 
                                        touch_scores=touch_scores[obj_id])
                    # if interaction[obj_id] >= 0:
                    #     second_obj_id = int(interaction[obj_id])
                    #     second_obj_bbox = pred_boxes[second_obj_id]
                    #     # print("\n\n\n\n")
                    #     # print("22222222222", obj_id)
                    #     # print(f"hand_id={count}\n"
                    #     #     f"hand_bbox={pred_boxes[second_obj_id]}\n"
                    #     #     f"hand_mask={pred_masks[second_obj_id]}\n"
                    #     #     f"contactState={parse_contact(int(contact_state[second_obj_id].item()))}\n"
                    #     #     f"hand_side={hand_side[second_obj_id].item()}\n"
                    #     #     f"touch_type={parse_touch(int(touch_type[second_obj_id].item()))}\n"
                    #     #     f"grasp={grasp[second_obj_id].item()}\n"
                    #     #     f"pred_score={pred_scores[second_obj_id]}\n"
                    #     #     f"grasp_scores={grasp_scores[second_obj_id]}")

                    #     is_duplicate = False
                    #     for processed_bbox in processed_objects:
                    #         if calculate_iou(second_obj_bbox, processed_bbox) > 0.7:  # IoU threshold
                    #             is_duplicate = True
                    #             break

                    #     if not is_duplicate:
                    #         curr_hand.set_second_obj(obj_bbox=second_obj_bbox, obj_masks=pred_masks[second_obj_id], 
                    #                                 pred_score= pred_scores[second_obj_id])
                    #         # crop_and_save_object(im, second_obj_bbox, save_dir, second_obj_id)
                    #         processed_objects.append(second_obj_bbox)

                hand_list.append(curr_hand)
   
        return self.process_hands_list(image, hand_list)   
    
       
    def process_hands_list(self, image, hand_list):
        
        def process_masked_image(image, mask):            
            if len(image.shape) == 3:  
                mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                image_masked = np.where(mask_3channel, image, 0)

                # Create an alpha channel: white (255) where mask is present, black (0) where it is absent
                alpha_channel = (mask * 255).astype(np.uint8)

                # Convert the image to RGBA
                rgba_image = np.dstack((image_masked, alpha_channel))

                return rgba_image  # This is now an RGBA image with transparency

            return None

        
        hand_interaction_info = {"num_of_hands": 0,
                                "which_hands": [],
                                "interact_with_same_object": False}
        
        for hand in hand_list:
            which_hand = hand.hand_side
            if which_hand not in hand_interaction_info:
                hand_interaction_info[which_hand] = {}
                hand_interaction_info['num_of_hands'] += 1
                hand_interaction_info['which_hands'].append(which_hand)
            else:
                continue
            
            hand_interaction_info[which_hand]['obj_bbox'] = hand.obj_bbox
            hand_interaction_info[which_hand]['obj_masks'] = hand.obj_masks
            hand_interaction_info[which_hand]['contactState'] = hand.contactState
            hand_interaction_info[which_hand]['obj_touch'] = hand.obj_touch
            hand_interaction_info[which_hand]['obj_image'] = crop_image_with_bbox(image, hand.obj_bbox) if hand.obj_bbox is not None else None

            if hand_interaction_info[which_hand]['obj_image'] is not None:
                cv2.imwrite(f"obj_image_{which_hand}.png", hand_interaction_info[which_hand]['obj_image'])
            if hand_interaction_info[which_hand]['obj_masks'] is not None:
                output_mask = process_masked_image(image, hand.obj_masks)
                output_mask = crop_image_with_bbox(output_mask, hand.obj_bbox)
                hand_interaction_info[which_hand]['obj_masks'] = output_mask
                cv2.imwrite(f"obj_mask_{which_hand}.png", output_mask)

        obj_bboxes = []
        obj_masks = []

        # Collect bounding boxes and masks
        for hand_side, info in hand_interaction_info.items():
            if hand_side == "left" or hand_side == "right":
                if info['obj_image'] is not None:
                    obj_bboxes.append(info['obj_bbox'])
                    obj_masks.append(info['obj_masks'])

        # Check for duplicates using IoU
        if len(obj_bboxes) > 1:
            iou, smaller_index = calculate_iou(obj_bboxes[0], obj_bboxes[1])
            if iou > 0.7:
                hand_interaction_info['interact_with_same_object'] = True

        draw_and_save_bounding_boxes(image, hand_interaction_info, "image_with_bounding_boxes.png")
        
        return hand_interaction_info
         