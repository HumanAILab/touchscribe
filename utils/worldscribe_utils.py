import base64
import numpy as np
import cv2
from collections import Counter
import pytesseract

def synthesize_hand_caption(sentences):
    hand_presence = {"left": False, "right": False}
    hand_actions = {"left": None, "right": None}
    hand_exited = {"left": False, "right": False}
    
    for sentence in sentences:
        if "your left hand just out" in sentence or "your left hand left my view" in sentence:
            hand_exited["left"] = True
            hand_presence["left"] = False
        elif "left hand" in sentence:
            hand_presence["left"] = True
            for action in ["touching", "grabbing", "pointing"]:
                if action in sentence:
                    hand_actions["left"] = action
        
        if "your right hand just out" in sentence or "your right hand left my view" in sentence:
            hand_exited["right"] = True
            hand_presence["right"] = False
        elif "right hand" in sentence:
            hand_presence["right"] = True
            for action in ["touching", "grabbing", "pointing"]:
                if action in sentence:
                    hand_actions["right"] = action
    
    if hand_exited["left"] and hand_exited["right"]:
        return "your both hands just out"
    
    output = []
    if hand_presence["left"] and hand_presence["right"]:
        if hand_actions["left"] == hand_actions["right"] and hand_actions["left"] is not None:
            output.append(f"I see your both hands are {hand_actions['left']}")
        else:
            output.append("I see your both hands")
            action_phrases = []
            if hand_actions["right"]:
                action_phrases.append(f"right hand {hand_actions['right']}")
            if hand_actions["left"]:
                action_phrases.append(f"left hand {hand_actions['left']}")
            if action_phrases:
                output[-1] += " with " + " and ".join(action_phrases)
    elif hand_presence["left"]:
        if hand_actions["left"]:
            output.append(f"I see your left hand is {hand_actions['left']}")
        else:
            output.append("I see your left hand")
    elif hand_presence["right"]:
        if hand_actions["right"]:
            output.append(f"I see your right hand is {hand_actions['right']}")
        else:
            output.append("I see your right hand")
    
    if hand_exited["left"] and not hand_exited["right"]:
        output.append("your left hand just out")
    if hand_exited["right"] and not hand_exited["left"]:
        output.append("your right hand just out")
    
    return " and ".join(output)

def is_point_inside_bbox(tip_location, bbox):
    """
    Check if a point (px, py) is inside the bounding box.

    :param px: x-coordinate of the point
    :param py: y-coordinate of the point
    :param bbox: Bounding box in the format [x, y, width, height]
    :return: True if the point is inside the bounding box, False otherwise
    """
    try:
        px, py = tip_location
        bx, by, bx2, by2 = bbox  # Extract bounding box coordinates

        output = bx <= px <= bx2 and by <= py <= by2
    except:
        return False
    return output

def base64_to_cv2_image(base64_string,Debug=False):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if Debug: cv2.imwrite("test.jpg", img)
    return img

def crop_mask_image(img, mask, box):
    box = box.astype(int)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), 
               interpolation = cv2.INTER_LINEAR)
    # mask = cv2.merge((mask,mask,mask)).astype(np.uint8)
    mask_not = cv2.bitwise_not(mask)[box[1]:box[3],box[0]:box[2]]
    # print(img.shape, mask.shape)
    # print(img.dtype, mask.dtype)
    img = cv2.bitwise_and(img, mask)
    output = img[box[1]:box[3],box[0]:box[2]] + mask_not
    return output

def crop_image(img, box, DEBUG=False):
    name = "test"
    box = box.astype(int)
    output = img[box[1]:box[3],box[0]:box[2]]
    if DEBUG: 
        cv2.imwrite(f"temp_out/{name}.jpg", output)
    return output


def get_caption_from_yolo_classes(cls_list):
    temp = list(set(cls_list))
    output = ""
    temp = sorted(temp)
    if len(temp) == 1: return temp[0]
    for i,cls in enumerate(temp):
        
        if i == len(temp)-1:
            output += " and " + cls 
        elif i==0:
            output += cls
        else:
            output += ", " + cls 
    return output


def encode_image_to_base64(numpy_image):
    _, encoded_image = cv2.imencode('.jpg', numpy_image)
    # Convert to base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    return base64_string


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate the union area by using the formula: union(A,B) = A + B - intersection(A,B)
    union_area = boxA_area + boxB_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou

def is_inside(A, B):
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    return Ax1 <= Bx1 and Ay1 <= By1 and Ax2 >= Bx2 and Ay2 >= By2

def find_most_overlapping_box(cls_boxes, text_box):
    max_iou = 80
    most_overlapping_pair = None

    for i in range(len(cls_boxes)):
            iou = calculate_iou(cls_boxes[i], text_box)
            print('iou', iou)
            if iou > max_iou:
                max_iou = iou
                most_overlapping_pair = i
            elif is_inside(cls_boxes[i], text_box):
                most_overlapping_pair = i
    return most_overlapping_pair


def is_symbols_only(s):
    # Check if each character in the string is not a letter, digit, or whitespace
    return all(not char.isalpha() and not char.isdigit() and not char.isspace() for char in s)


def get_texts(frame,conf=80):
    data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    # print(data)
    output_list = []
    for i,text in enumerate(data['text']):
        if int(data['conf'][i]) > conf:  # You can adjust the confidence level
            if text.isspace() or is_symbols_only(text): continue
            # print('hello' , text,i, data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            temp = {}            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            temp['text'] = text
            temp['box'] = [x , y , x+w , y+h]
            output_list.append(temp)
            print(f'Text: {text}, Position: ({x}, {y}), Size: ({w}x{h})')
    
    if len(output_list) == 0: return None

    return output_list



def get_caption_from_yolo_classes_adv(frame_info, object_preference=None):
    if len(frame_info['object_classes']) ==0 and frame_info['text'] is None: return None
    elif len(frame_info['object_classes']) ==0 and len(frame_info['text']) > 0:
        output = "There are texts showing "
        for i,t in enumerate(frame_info['text']):
            print(t)
            if i == len(frame_info['text']) -1: output += t['text'] + "."
            output  += t['text'] + ", "
        return output

    cls_list = frame_info['object_classes']
    boxes = frame_info['boxes']
    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]

    object_constraint = []
    if object_preference: 
        object_constraint = [item for item in object_preference if object_preference[item] == 'false']
    indexes  = [i for i,item in enumerate(cls_list) if item not in object_constraint]
    cls_list = [cls_list[i] for i in indexes]
    areas    = [areas[i] for i in indexes]
    boxes    = [boxes[i] for i in indexes]


    
    text_info = frame_info['text']

    # print(text_info)

    output_text_sentence = ""

    if text_info is not None:
        cls_w_text = [None for i in boxes]
        for ti in text_info:
            text = ti['text']
            tbox = ti['box']
            index = find_most_overlapping_box(boxes, tbox)
            # print('index', index)
            if index is None: continue
            if cls_w_text[index] is None: cls_w_text[index] = []
            cls_w_text[index].append(text)
        
        
        for i,c in enumerate(frame_info['object_classes']):
            if cls_w_text[i] is not None:
                text = ""
                for t in cls_w_text[i]:
                    text +=t+' '
                
                output_text_sentence += f"There are texts on {c} showing {text.strip()}. "

        print(cls_w_text)

    temp  = list(set(cls_list))
    area_temp = []
    for cls in temp:
        temporay_area = 0
        for i,a in enumerate(areas):
            if cls_list[i] == cls:
                if a > temporay_area:
                    temporay_area = a
        area_temp.append(temporay_area)
    count = Counter(cls_list)

    sorted_pairs = sorted(zip(area_temp, temp), reverse=True)

    # Extract the sorted items from the sorted pairs
    temp = [item for _, item in sorted_pairs]

    output = ""
    # temp = sorted(temp)
    if len(temp) == 1: return "A "+ temp[0] + output_text_sentence
    for i,cls in enumerate(temp):
        
        if i == len(temp)-1:
            output += " and " + f"{count[cls] if count[cls] >1 else 'a'} " + cls 
        elif i==0:
            output += f"{count[cls] if count[cls] >1 else 'a'} " + cls 
        else:
            output += ", " + f"{count[cls] if count[cls] >1 else 'a'} " + cls 

    # if len(frame_info['text'])>0:
    #     output += f" and there is text: {frame_info['text']}."
    return output + " "+  output_text_sentence








# frame_info = {}
# frame_info['object_classes']=['laptop', 'desk']
# frame_info['ids'] =[1, 2]

# frame_info['boxes']= [[     71.109   ,   26.222    ,  357.38    ,  421.71],
#  [          0    ,  140.79    ,  356.76   ,      638]]
# import cv2
# image = cv2.imread('stream.jpg')
# frame_info['frame'] = image
# frame_info['text']  = get_texts(image)





# for box in frame_info['boxes']:
#     print(box)
#     image = cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255, 0, 0) , 2) 

# for text in frame_info['text']:
#     box = text['box']
#     image = cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255, 0, 0) , 2) 


# cv2.imwrite('Image1234.jpg', image)  


# print(get_caption_from_yolo_classes_adv(frame_info))
