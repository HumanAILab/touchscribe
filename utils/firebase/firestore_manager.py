"""firestore manage class"""
import os
import time
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
# Initialize the app with a service account
if not firebase_admin._apps:
    cred = credentials.Certificate(os.environ.get('FIREBASE_CREDENTIALS_PATH', 'firebase-adminsdk.json'))
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', 'https://your-project-default-rtdb.firebaseio.com/')
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
    