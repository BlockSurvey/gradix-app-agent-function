from fastapi import FastAPI

import config
from database.firebase_db import firestore_client
from controllers.user_controller import UserController

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    # Create a new instance of UserController
    user_controller = UserController()
    return user_controller.get_user(user_id)

@app.get("/")
def read_root():
    # return the status
    return {"status": "ok"}