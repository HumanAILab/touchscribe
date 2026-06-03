# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db

# # Initialize the app with a service account
# if not firebase_admin._apps:
#     cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
#     })
# import traceback


# def listening(event):
#     try:
#         msg = event.data
#         play_audio()
#         print(event.data)

#     except Exception as e:
#         print("Errors: ", e)
#         traceback.print_exc()  # This will print the full traceback

#     return

import sounddevice as sd
import soundfile as sf

def play_audio(event):

    if event == "hands_detected": filename = "audio/softnotif.wav"
    elif event == "hands_left": filename = "audio/chime.wav"
    # if event_name == "new_scene":
    #     filename = f'audio/new.wav'

    # elif event_name == "long_stay_scene":
    #     filename = f'audio/long.wav'

    # elif event_name == "empty_scene":
    #     filename = f'audio/empty.wav'

    # elif event_name == "messy_scene":
    #     filename = f'audio/messy.wav'

    # elif event_name == "empty_long_scene":
    #     filename = f'audio/empty_long.wav'

    # elif event_name == "generate_message":
    #     filename = f'audio/softnotif.wav'

    # else:
    #     filename = f'audio/softnotif.wav'
    if filename:

        data, fs = sf.read(filename, dtype='float32')  # 'fs' is the sampling rate

        # Play the audio
        sd.play(data, fs)
        sd.wait() 
    return
# ref = db.reference('audio_player/')
# ref.listen(listening)