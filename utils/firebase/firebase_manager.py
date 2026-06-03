"""firebase_manager"""
import traceback
import ast
import time
import zlib 
import json
import base64
from firebase_admin import db
from utils.worldscribe_utils import base64_to_cv2_image
from utils.spatial import get_direction_by_degree

STOP_DEGREE = 2

class FirebaseWriteManager():
    def __init__(self):
        self.ref_manipulation_table = db.reference('system_query/manipulation_table')
        self.ref_caption = db.reference('caption')
        self.ref_caption_buffer = db.reference('caption_buffer/')
        self.ref_caption_buffer.set("start")

        self.ref_audio_player = db.reference('audio_player/')

        self.ref_moondream_caption_buffer = db.reference('moondream_buffer/')
        self.ref_moondream_caption_buffer.set("start")


        self.ref_user_degree = db.reference('true_heading/')
        self.ref_image_data = db.reference('image_data/')
        self.ref_handscribe_image_data = db.reference('ref_handscribe_image_data/')
        self.ref_memory_image_data = db.reference('memory_image_data/')
        self.ref_agent_response = db.reference('agent_response/')
        self.ref_stop_streaming = db.reference('stop_streaming/')
        self.ref_swipe_data = db.reference('swipe_data/')
        self.ref_user_query_buffer = db.reference('user_query_buffer/')
        self.ref_new_object_check_image = db.reference('new_object_check_image/')
        self.ref_agent_ending_msg = db.reference('agent_ending_msg/')
        self.ref_text_reading_msg = db.reference('text_reading/')
        self.lock_caption_buffer = False
        self.is_ranking = False

    def send_new_object_check_image(self, packet):
        self.ref_new_object_check_image.set(packet)
        return

    def send_user_query_to_remote(self, packet):
        self.ref_user_query_buffer.set(packet)
        return

    def send_thumb_gesture(self, packet):
        self.ref_swipe_data.set(packet)
        return
    
    def send_image_to_server(self, packet):
        # compressed_packet = self.compress_data(packet)
        self.ref_image_data.set(packet)
        return 
    
    def send_image_to_handscribe_server(self, packet):
        # compressed_packet = self.compress_data(packet)
        self.ref_handscribe_image_data.set(packet)
        return
    
    def send_memory_image_to_server(self, packet):
        # compressed_packet = self.compress_data(packet)
        self.ref_memory_image_data.set(packet)
        return 
    
    def send_agent_response(self, msg, action):
        try:
            self.ref_agent_response.set({
                'timestamp': int(time.time()),
                'caption': msg,
                'action': action
            })
        except Exception as e:
            print("error: ", e)
        return
    
    def send_agent_ending_msg(self, msg, action):
        try:
            self.ref_agent_ending_msg.set({
                'timestamp': int(time.time()),
                'caption': msg,
                'action': action
            })
        except Exception as e:
            print("error: ", e)
        return
    
    def send_text_reading(self, msg):
        try:
            self.ref_text_reading_msg.set({
                'timestamp': int(time.time()),
                'caption': msg,
                'action': 'none'
            })
        except Exception as e:
            print("error: ", e)
        return
    
    
    def send_stop_streaming(self, msg):
        try:
            self.ref_stop_streaming.set({
                'timestamp': int(time.time()),
                'stop_command': msg,
            })
        except Exception as e:
            print("error: ", e)
        return

    def play_audio(self,event_name):
        self.ref_audio_player.set(event_name)
        return
    

    def update_caption(self, caption):
        try:
            self.ref_caption.set({
                'timestamp': int(time.time()),
                'text':caption
            })
        except Exception as e:
            print("error: ", e)

    def reset_vit_caption_buffer(self):
        self.ref_vit_caption_buffer.set("start")


    def get_moondream_caption_buffer(self):
        # self.lock_caption_buffer = True
        # while not self.lock_blip2_caption_buffer:
        values = self.ref_moondream_caption_buffer.get()
        if values == "start" or values == None: return "start"
        values = list(self.ref_moondream_caption_buffer.get())
        return values
        

    def reset_caption_buffer(self):
        self.ref_caption_buffer.set("start")

    def update_caption_buffer(self, queue_list):
        self.lock_caption_buffer = True
        while self.lock_caption_buffer:
            self.ref_caption_buffer.set(queue_list)
            # for item in queue_list:
                # self.ref_caption_buffer.push(item)
            self.lock_caption_buffer = False

    def get_caption_buffer(self):
        # self.lock_caption_buffer = True
        while not self.lock_caption_buffer:
            values = self.ref_caption_buffer.get()
            if values == "start" or values == None: return "start"
            # print("self.ref_caption_buffer.get()", self.ref_caption_buffer.get())
            values = list(self.ref_caption_buffer.get())
            # print(values)
            return values
            # self.lock_caption_buffer = False
        
    def get_user_clock(self):
        degree = float(self.ref_user_degree.get())
        return get_direction_by_degree(degree)
    
    def get_user_degree(self):
        degree = float(self.ref_user_degree.get())
        return degree
    

def angle_difference(prev, current):
    # Calculate the direct difference
    direct_diff = abs(current - prev)
    
    # Calculate the circular difference
    circular_diff = 360 - direct_diff
    
    # Return the smallest difference
    return min(direct_diff, circular_diff)


class FirebaseManager:

    def __init__(self, data_processor, image_queue=None):

        self.data_processor = data_processor

        self.ref_audio_data_start = True

        # self.ref_audio_data = db.reference('audio_data/')
        # self.ref_audio_data.listen(self.listener_audio_data)
        

        self.ref_image_data_start = True
        self.ref_image_data = db.reference('image_data/')

        self.ref_caption_buffer_start = True
        self.ref_caption_buffer = db.reference('caption_buffer/')

        self.ref_object_category_start = True
        self.ref_object_category = db.reference('object_categories/')
        self.ref_object_category.listen(self.lister_object_category_data)

        self.image_queue = image_queue

        self.image_present_time = time.time()

        self.user_prev_degree = None
        self.user_degree_history = []
        self.ref_user_degree_start = True
        self.ref_user_degree = db.reference('true_heading/')
        self.ref_user_degree.listen(self.lister_user_degree_data)

        self.ref_user_name_start = True
        self.ref_user_name = db.reference('user_name/')
        self.ref_user_name.listen(self.listen_user_name)

        self.ref_is_streaming_start = True
        self.ref_is_streaming = db.reference('is_streaming/')
        self.ref_is_streaming.listen(self.listen_is_streaming)

        self.ref_object_text_start = True
        self.ref_object_text = db.reference('object_text/')
        self.ref_object_text.listen(self.listener_object_text_data)

        self.ref_new_object_check_start = True
        self.ref_new_object_check = db.reference('new_object_check/')
        self.ref_new_object_check.listen(self.listener_new_object_check_data)

        self.ref_text_system_message_start = True
        self.ref_text_system_message = db.reference('text_system_message/')
        self.ref_text_system_message.listen(self.listener_text_system_message)


    def listener_text_system_message(self, packet):
        if self.ref_text_system_message_start:
            self.ref_text_system_message_start = False
            print("******************************")
            print("Listening to text_system_message")
            print("******************************")
            return 
        data = dict(packet.data)
        timestamp           = data['timestamp']
        timestamp_s         = data['timestamp_s']
        text_system_message = data['text']
        if self.data_processor.current_caption_source == "object_text":
            return
        self.data_processor.text_system_message.append(data)


    def listener_new_object_check_data(self, packet):
        if self.ref_new_object_check_start:
            self.ref_new_object_check_start = False
            print("******************************")
            print("Listening to new_object_check")
            print("******************************")
            return 
        data = dict(packet.data)
        timestamp   = data['timestamp']
        timestamp_s = data['timestamp_s']
        complete    = data['complete']
        is_new_object_on_left = data['is_new_object_on_left']
        is_new_object_on_right = data['is_new_object_on_right']

        self.data_processor.is_new_object_on_left = is_new_object_on_left
        self.data_processor.is_new_object_on_right = is_new_object_on_right
        self.data_processor.new_object_check = complete


    def listener_object_text_data(self, packet):
        if self.ref_object_text_start:
            self.ref_object_text_start = False
            print("******************************")
            print("Listening to object_text_data")
            print("******************************")
            return 
        print("text appended \n "*100)
        data = dict(packet.data)
        self.data_processor.object_text_buffer.append(data)
        
        
    def listen_is_streaming(self, event):
        
        try:
            if self.ref_is_streaming_start:
                self.ref_is_streaming_start = False
            else:
                print(event.data)
                # user_name = str(event.data)
                if event.data:
                    data = dict(event.data)
                    command = data['input']
                    if command == 'unlock': 
                        print('unlock\n'*100)
                        self.data_processor.reset_all_text_buffer()
                        self.data_processor.ttsManager.unlock_streaming()

                    self.data_processor.ttsManager.is_streaming = False

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback


        return    
    

    def listen_user_name(self,event):
        
        try:
            if self.ref_user_name_start:
                self.ref_user_name_start = False
            else:
                print(event.data)
                user_name = str(event.data)
                self.data_processor.firestoreManager.set_name(user_name)
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback


        return

    def lister_user_degree_data(self, event):
        try:
            if self.ref_user_degree_start:
                self.ref_user_degree_start = False
            else:
                if self.user_prev_degree is None: 
                    self.user_prev_degree = float(event.data)
                else:
                    user_current_degree = float(event.data)
                    self.data_processor.user_current_degree = user_current_degree
                    print("user_current_degree: ", self.user_prev_degree, user_current_degree)
                    self.data_processor.user_is_moving = False
                    diff = angle_difference(user_current_degree, self.user_prev_degree)
                    if diff >= STOP_DEGREE:
                        self.user_prev_degree = user_current_degree
                        self.data_processor.user_is_moving = True
                        print("***********is moving***********\n"*10)

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return
    
    
    
    def lister_object_category_data(self, event):
        try:
            if self.ref_object_category_start:
                self.ref_object_category_start = False
            else:
                print(event.data)

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return


    def lister_caption_buffer_data(self, event):
        try:
            if self.ref_caption_buffer_start:
                self.ref_caption_buffer_start = False
            else:
                print(event.data)

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return

    def listener_image_data(self, event):
        # print(event.event_type)  # can be 'put' or 'patch'
        # print(event.path)  # relative to the reference, it seems
        # print(event.data)  # new data at /reference/event.path. None if deleted
        try:
            if self.ref_image_data_start:
                self.ref_image_data_start = False
            else:
                # print(str(event.data))
                # print(time.time())
                # if time.time() - self.image_present_time > 1:
                self.ref_send_image.set("false")
                self.image_present_time = time.time()
                s = event.data
                image = s['base64']
                timestamp = s['timestamp']
                frame = base64_to_cv2_image(image)
                if self.image_queue: self.image_queue.put(("img_name",frame))
                self.data_processor.process_data_by_yolo(frame, {'frame':frame, 'timestamp':timestamp})
                self.ref_send_image.set("true")

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc() 


    # def listener_audio_data(self, event):
    #     # print(event.event_type)  # can be 'put' or 'patch'
    #     # print(event.path)  # relative to the reference, it seems
    #     # print(event.data)  # new data at /reference/event.path. None if deleted
    #     try:
    #         if self.ref_audio_data_start:
    #             self.ref_audio_data_start = False
    #         else:
    #             s = str(event.data)
    #             s = s.replace("\'", "\"")
    #             dict_obj = ast.literal_eval(s)
    #             self.data_processor.process_audio_data(dict_obj)
    #     except Exception as e:
    #         print("Errors: ", e)
    #         traceback.print_exc()  
    
    
if __name__ == '__main__':
    test1 = FirebaseManager(None)

    test = FirebaseWriteManager()
    queue_list = [{'caption': 'a black digital camera next to a black case', 'frame_id': 182, 'event': 'new_scene'}, {'caption': 'a large bag of organic whole cashews sits on a table', 'frame_id': 273, 'event': 'empty_scene'}, {'caption': 'the table is made of light brown wood', 'frame_id': 273, 'event': 'empty_scene'}, {'caption': 'near the bottom right corner is a blue object with white', 'frame_id': 273, 'event': 'empty_scene'}, {'caption': 'the backdrop includes a white surface and darker areas', 'frame_id': 273, 'event': 'empty_scene'}, {'caption': 'a gray shelf holds various items', 'frame_id': 262, 'event': 'new_scene'}, {'caption': 'a white charger and black cables are entangled atop the shelf', 'frame_id': 262, 'event': 'new_scene'}, {'caption': "the wall is white and there's a gray window shade", 'frame_id': 262, 'event': 'new_scene'}]
    test.update_caption_buffer(queue_list)