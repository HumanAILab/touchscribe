import io
import os
import requests
import numpy as np
import simpleaudio as sa
import tempfile
import subprocess
import sounddevice as sd
from functools import partial
import time
import threading  # Import threading for handling asynchronous volume changes

subscription_key = os.environ.get("AZURE_SPEECH_KEY", "")
region = os.environ.get("AZURE_SPEECH_REGION", "eastus")  # For example, "westus"

# The TTS API endpoint
tts_uri = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

# Headers for the request
headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/ssml+xml",
    "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",  # Audio output format
}

# The SSML body, wrapping your text in speech synthesis markup language
# ssml = """
# <speak version='1.0' xml:lang='en-US'>
#     <voice xml:lang='en-US' xml:gender='Female' name='en-US-JennyNeural'>
#         Your text goes here.
#     </voice>
# </speak>
# """
test_string = "Hello world! This is a streaming test. And I would like to test if the volume will be decreased. "


class TTSManager:

    def __init__(self, firebaseWriteManager, model="tts-1", voice="onyx"):
        self.volume_factor = 0.1  # Default volume
          # Default speed is 1.0 (normal speed)
        self.model = model
        self.voice = voice
        self.response_format = "opus"
        self.lock = threading.Lock()  # Use a lock to safely update volume_factor from different threads
        self.firebaseWriteManager = firebaseWriteManager
        self.is_streaming = False

        self.low_volume = False
        self.high_volume = False

        self.stream_locked = False

        self.normal_speed = 1.4
        self.slow_speed = 1.1
        self.speech_speed = self.normal_speed
        self.user_current_clock = 0


    def get_speed_text(self):
        string = str(self.speech_speed*100)
        return string

    def set_volume(self, volume):
        with self.lock:  # Ensure thread-safe updates to volume_factor
            self.volume_factor = max(0, min(volume, 1))  # Clamp volume between 0 and 1
    
    def stop_streaming(self):
        with self.lock:
            self.stream_locked = True
            self.is_streaming = False


    def start_streaming(self):
        with self.lock:
            self.stream_locked = False
            # self.is_streaming = False

    def reset_streaming(self):
        with self.lock:
            self.stream_locked = True
            time.sleep(0.05)
            self.stream_locked = False
            # self.is_streaming = False

    def check_volume_state(self):
        if self.volume_factor < 0.5:
            return "low"
        elif self.volume_factor == 0.5:
            return "middle"
        elif self.volume_factor > 0.5:
            return "high"
        
    def decrease_volume_and_back(self):
        # with self.lock:  # Ensure thread-safe updates to volume_factor
        if not self.low_volume:
            self.low_volume = True
            time.sleep(0.1)
            self.set_volume(0.4)  
            time.sleep(0.1)
            self.set_volume(0.3)  
            time.sleep(0.1)
            self.set_volume(0.2)  
            time.sleep(0.1)
            self.set_volume(0.1)  
        else:
            time.sleep(0.1)
            self.set_volume(0.2)  
            time.sleep(0.1)
            self.set_volume(0.3)  
            time.sleep(0.1)
            self.set_volume(0.4)  
            time.sleep(0.1)
            self.set_volume(0.5)  
            self.low_volume = False
        return
    
    def increase_volume_and_back(self):
        # with self.lock:  # Ensure thread-safe updates to volume_factor
        if not self.high_volume:
            self.high_volume = True
            time.sleep(0.1)
            self.set_volume(0.3)  
            time.sleep(0.1)
            self.set_volume(0.5)  
            time.sleep(0.1)
            self.set_volume(0.7)  
            time.sleep(0.1)
            self.set_volume(1.0)
        else:
            self.set_volume(0.9)  
            time.sleep(0.1)
            self.set_volume(0.7)  
            time.sleep(0.1)
            self.set_volume(0.5)  
            time.sleep(0.1)
            self.set_volume(0.3) 
            time.sleep(0.1)
            self.set_volume(self.volume_factor)  
            self.high_volume = False
        return
    
    def increase_volume_and_back_mild(self):
        # with self.lock:  # Ensure thread-safe updates to volume_factor
        if not self.high_volume:
            self.high_volume = True
            time.sleep(0.1)
            self.set_volume(0.3)  
            time.sleep(0.1)
            self.set_volume(0.4)  
            time.sleep(0.1)
            self.set_volume(0.5)
        else:
            self.set_volume(0.4)  
            time.sleep(0.1)
            self.set_volume(0.3) 
            time.sleep(0.1)
            self.set_volume(self.volume_factor)  
            self.high_volume = False
        return

    def set_speech_speed(self, speed):
        # with self.lock:
        self.speech_speed = max(0.5, min(speed, 2.0))  # Limit speed to a reasonable range (e.g., 0.5x to 2x)


    def is_user_turn(self):
        clock = self.firebaseWriteManager.get_user_clock()
        if self.user_current_clock==0:
            self.user_current_clock = clock
            return False
        
        elif self.user_current_clock != clock:
            self.user_current_clock = clock
            return True
        
        else: 
            return False
        

    def live_tts(self, input_string=None):
        if input_string is None: return
        # print("Starting TTS...")
        # AndrewNeural AvaNeural
        ssml = f"""
                <speak version='1.0' xml:lang='en-US'>
                    <voice xml:lang='en-US' xml:gender='Female' name='en-US-AvaNeural'>
                        <prosody rate='{self.speech_speed}'>
                            {input_string}
                        </prosody>
                    </voice>
                </speak>
                """
        print(ssml)
        # ssml = ssml.format(input_text=input_string, speed=self.get_speed_text())
        with requests.post(tts_uri, headers=headers, data=ssml) as response:
            print("response.status_code", response.status_code)
            if response.status_code == 200:
                self.is_streaming = True
                opus_buffer = io.BytesIO()
                for chunk in response.iter_content(chunk_size=4096):
                    opus_buffer.write(chunk)
                opus_buffer.seek(0)

                process = subprocess.Popen(['ffmpeg', '-y', '-i', 'pipe:0', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', '-f', 'wav', 'pipe:1'],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                wav_data, _ = process.communicate(input=opus_buffer.getvalue())

                wav_buffer = io.BytesIO(wav_data)
                trim_time_seconds = 0.01  # Time to trim from the start
                bytes_per_frame = 2  # 16-bit audio
                sample_rate = 44100  # Assumed sample rate
                adjusted_samplerate = int(sample_rate * 1.0)

                # Calculate the number of bytes to skip
                bytes_to_skip = int(sample_rate * bytes_per_frame * trim_time_seconds)
                wav_buffer.seek(bytes_to_skip)
                # wav_buffer.seek(0)  # Go to the start of the buffer

                def audio_callback(outdata, frames, _time, status):
                    
                    with self.lock:  # Use the lock to ensure thread-safe access to volume_factor
                    
                        data = wav_buffer.read(frames * 2)  # 2 bytes per frame
                        decoded_data = np.frombuffer(data, dtype=np.int16) if data else np.array([], dtype=np.int16)

                        if len(decoded_data) < frames:
                            padding = np.zeros((frames - len(decoded_data),), dtype=np.int16)
                            decoded_data = np.concatenate((decoded_data, padding))
                            # time.sleep(1)
                            self.is_streaming = False
                            raise sd.CallbackStop

                        decoded_data = (decoded_data * self.volume_factor).reshape(outdata.shape)
                        outdata[:] = decoded_data

                # adjusted_samplerate = int(48000 * 1.0)

                stream = sd.OutputStream(callback=audio_callback, dtype=np.int16, channels=1, samplerate=adjusted_samplerate)
                if self.stream_locked: 
                    # print("732647863287568376587623487623876832768273684")
                    # print("732647863287568376587623487623876832768273684")
                    # print("732647863287568376587623487623876832768273684")
                    # print("732647863287568376587623487623876832768273684")
                    print("")
                    self.is_streaming = False
                    return
                with stream:
                    # self.is_streaming = True
                    stream.start()
                    print("")
                    while self.is_streaming and not self.stream_locked:
                        # test = self.stream_locked
                        time.sleep(0.001)   # Use input to wait for user action or implement another stopping condition

if __name__ == '__main__':
    ttsManager = TTSManager(None)
    start_found = False
    threading.Thread(target=ttsManager.live_tts, args=(test_string,)).start()

    time.sleep(3)
    ttsManager.increase_volume_and_back()
    # Example code to stop streaming after 5 seconds
    # time.sleep(3)
    # ttsManager.stop_streaming()