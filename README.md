<!-- Install dependencies -->
pip install -r requirements.txt

<!-- Start Command -->
<!-- For development -->
uvicorn app:app --reload

<!-- To run using specific python version -->
python3.11 -m uvicorn app:app --reload

<!-- To run in the virtual environment -->
python3 -m venv env
source env/bin/activate
pip freeze > requirements.txt
deactivate

<!-- For production -->
uvicorn app:app --host 0.0.0.0 --port 8080