import firebase_admin
from firebase_admin import credentials, firestore

from config import FIREBASE_SERVICE_ACCOUNT_KEY_LOCATION

cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_LOCATION)
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore service
firestore_client = firestore.client()