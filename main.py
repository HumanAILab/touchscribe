import socket
import threading
import json
import queue
import cv2
import os
import firebase_admin
import base64
from datetime import datetime
import time
import numpy as np
import uuid

from firebase_admin import credentials

if not firebase_admin._apps:
    cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
    })

from utils.point_cloud_utils import save_colorized_point_cloud
from utils.data_processor import DataProcessor
from utils.worldscribe_utils import base64_to_cv2_image
from utils.firebase.firebase_manager import FirebaseManager
from config.config_server import SERVER_IP, VISUAL_SERVER_PORT, DEPTH_SERVER_PORT

# TCP protocol setup
recv_packet_size = 65535
packet_tail = "_TAIL".encode('utf-8') # this is the tail data that identifies the end of the packet which is the string _TAIL
count=0



def process_server_visual_data(data):
    # decode the data string from smartphone, which is a json
    server_data_dict_data_str = data.decode("utf-8")
    # decode the json into dictionary 
    server_data_dict = json.loads(server_data_dict_data_str)
    # get the image frame in the form of base64string
    frame_base64string = server_data_dict["camera_data"]['rgb_frame']
    # convert the base64 to the cv2 image
    frame = base64_to_cv2_image(frame_base64string)
    server_data_dict["camera_data"]['rgb_frame'] = frame
    server_data_dict['frame_base64'] = frame_base64string
    # global count
    # count+=1
    cv2.imwrite(f"image_data/{time.time()}.jpg", frame)

    # print("frame.shape: ", frame.shape)

    # This is the gobal dataProcessor, where we only have one in our system
    global dataProcessor

    # worldscribe
    # frame = dataProcessor.process_data_by_yolo(frame, server_data_dict)

    frame = dataProcessor.process_data_by_hands(frame, server_data_dict)

    # check if the frame exists
    if type(frame) is type(None): return

    # put the frame into a image queue for display the image using function display_image below
    global image_queue
    image_queue.put(("img_name", frame))
    return

def process_depth_data(data):

    server_data_dict_data_str = data.decode("utf-8")
    server_data_dict = json.loads(server_data_dict_data_str)

    # depth_base64string = server_data_dict["camera_data"]['depth_frame_string']
    # height             = server_data_dict["camera_data"]['height']
    # width              = server_data_dict["camera_data"]['width']
    # fx                 = server_data_dict["camera_data"]['fx']
    # fy                 = server_data_dict["camera_data"]['fy']
    # cx                 = server_data_dict["camera_data"]['cx']
    # cy                 = server_data_dict["camera_data"]['cy']
    # camera_pose        = server_data_dict["camera_data"]['camera_pose']
    rgb_base64string   = server_data_dict["camera_data"]['rgb_frame_string']
    rgb_frame          = base64_to_cv2_image(rgb_base64string)

    # cv2.imwrite('frame.png', rgb_frame)
    global dataProcessor
    rgb_frame = dataProcessor.process_data_by_yolo(rgb_frame, server_data_dict)
    if type(rgb_frame) is type(None): return
    global image_queue
    image_queue.put(("img_name", rgb_frame))

    memory_storation_with_phone_info(server_data_dict)
    return


def memory_storation_with_phone_info(packet):
    
    timestamp          = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_s        = time.time()
    frame_id           = dataProcessor.frame_id
    _uuid               = str(uuid.uuid4().hex)
    description        = dataProcessor.current_caption
    description_source = dataProcessor.current_caption_source
    object_classes     = dataProcessor.frame_info['object_classes']
    ids                = dataProcessor.frame_info['ids']
    fx                 = packet['camera_data']['fx']
    fy                 = packet['camera_data']['fy']
    cx                 = packet['camera_data']['cx']
    cy                 = packet['camera_data']['cy']
    camera_pose        = packet['camera_data']['camera_pose']
    width              = packet['camera_data']['width']
    height             = packet['camera_data']['height']
    rgb_frame_string   = packet['camera_data']['rgb_frame_string']
    depth_frame_string = packet['camera_data']['depth_frame_string']
    folder             = "memory_data/" + "room_Feb18"
    
    # print(camera_pose)
    os.makedirs(folder, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(folder, str(i))):
        i += 1

    subfolder = os.path.join(folder, str(i))
    os.makedirs(subfolder, exist_ok=True)
    
    # Convert and save RGB image
    rgb_frame = base64_to_cv2_image(rgb_frame_string)
    high_res_rgb_frame = rgb_frame
    rgb_image_path = os.path.join(subfolder, "rgb_image.png")
    high_res_rgb_image_path = os.path.join(subfolder, "high_res_rgb_image.png")
    cv2.imwrite(high_res_rgb_image_path, high_res_rgb_frame)
    
    # Convert and save depth image
    depth_bytes = base64.b64decode(depth_frame_string)
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
    depth_array = np.rot90(depth_array, k=-1)  # Rotate 90 degrees clockwise
    depth_array = np.ascontiguousarray(depth_array)

    depth_array_path = os.path.join(subfolder, "depth_array.npy")
    np.save(depth_array_path, depth_array)
    
    depth_grayscale = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb_frame_resized = cv2.resize(rgb_frame, (depth_grayscale.shape[1], depth_grayscale.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth_image_path = os.path.join(subfolder, "depth_image.png")
    cv2.imwrite(depth_image_path, depth_grayscale)
    cv2.imwrite(rgb_image_path, rgb_frame_resized)

    # Save colorized point cloud
    point_cloud_path = os.path.join(subfolder, "point_cloud.ply")
    save_colorized_point_cloud(rgb_frame_resized, depth_array, packet['camera_data'], point_cloud_path)
    
    json_path = os.path.join(subfolder, "metadata.json")            
    
    # Save metadata
    packet_metadata = {
        "uuid": _uuid,
        "frame_id": frame_id,
        "date": timestamp,
        "timestamp_s": timestamp_s,
        "object_classes": object_classes,
        "ids": ids,
        
        "description": description,
        "description_source": description_source,
        
        "high_res_rgb_image_path": high_res_rgb_image_path,
        "rgb_image_path": rgb_image_path,
        "depth_image_path": depth_image_path,
        "depth_array_path": depth_array_path,
        "point_cloud_path": point_cloud_path,
        "json_data_path": json_path,
        
        "camera_data": json.dumps({
                                    "fx": fx,
                                    "fy": fy,
                                    "cx": cx,
                                    "cy": cy,
                                    "camera_pose": camera_pose
                                })
    }        
    
    with open(json_path, "w") as json_file:
        json.dump(packet_metadata, json_file, indent=4)
    
    print(f"===Descriptions: {description}")
    print(f"===Saved packet data in {subfolder}\n\n")
            

def display_image():
    while True:
        try:
            global image_queue
            # get the image from image_queue and display
            img_name, img = image_queue.get(block=True, timeout=.1)  # poll every 0.1 seconds
            cv2.imshow('Image preview', img)
        except queue.Empty:
            ...  # no new image to display
        key = cv2.pollKey()  # non-blocking
        if key & 0xff == ord('q'):
            cv2.destroyAllWindows()
            return
    

def handle_client(client_socket, addr, mode):
    data = b''
    while True:
        # receive data from the client smartphone   
        data += client_socket.recv(recv_packet_size)
        if data[-len(packet_tail):] == packet_tail:
            # check if this is the end of the data stream by identifying _TAIL
            data = data[:-len(packet_tail)]

            if mode == "depth":
                process_depth_data(data)

            elif mode == "visual":
                process_server_visual_data(data)

            # process_server_visual_data(data)
            data = b''

            # respond to the smartphone with this msg
            response = "ImageProccessed"
            client_socket.send(response.encode("utf-8"))

def run_server(server_ip = "127.0.0.1", port = 8000, mode="visual"):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to the host and port
        server.bind((server_ip, port))
        # listen for incoming connections
        server.listen()
        print(f"Listening on {server_ip}:{port}")

        while True:
            # accept a client connection
            client_socket, addr = server.accept()
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            # start a new thread to handle the client
            thread = threading.Thread(target=handle_client, args=(client_socket, addr, mode,))
            thread.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.close()

if __name__ == '__main__':
    image_queue = queue.Queue() # for display the frames for debugging purposes
    point_cloud_queue = queue.Queue() # for display the frames for debugging purposes
    dataProcessor = DataProcessor() # main data processor
    firebaseManager = FirebaseManager(dataProcessor, image_queue) # the gobal firebasemanager
    worldscribeServer = threading.Thread(target=run_server, args=(SERVER_IP, VISUAL_SERVER_PORT,)) # open a thread for server in order to make the display_image() runnable below
    worldscribeServer.start()
    
    # mageServer = threading.Thread(target=run_server, args=(SERVER_IP, DEPTH_SERVER_PORT, "depth"))
    # mageServer.start()

    display_image()