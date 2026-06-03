"""firestore manage class"""
import time
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
# Initialize the app with a service account
if not firebase_admin._apps:
    cred = credentials.Certificate('/home/rueiche/worldscribe/soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
    })
firestore_db = firestore.client()

class FirestoreManager:
    def __init__(self, name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.project = "handscribe"
        self.scene = "scene1"
        self.name = name
        self.user_ref = firestore_db.collection(self.project).document(self.name)
    
    def set_name(self, project, name):
        self.project = project
        self.name = name
        self.user_ref = firestore_db.collection(project).document(name)
        return
    
    def handscribe_update(self, caption_info):
        handscribe_refs = self.user_ref.collection('handscribe_log').document()
        handscribe_refs.set(caption_info)
        return

    def read_gpt_log(self, collection_name):
        for doc in self.user_ref.collection(collection_name).stream():
            print(f'{doc.id} => {doc.to_dict()}')
        return

if __name__ == '__main__':
    firestoreManager = FirestoreManager('rueiche')
    firestoreManager.handscribe_update({'caption':'hello world'})
    