import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# LLM configs
LLM_MODEL_GTP_3_5_TURBO = os.environ.get('LLM_MODEL_GTP_3_5_TURBO')
LLM_MODEL_GTP_4_O = os.environ.get('LLM_MODEL_GTP_4_O')

# Set the default and fallback LLM models
LLM_MODEL_DEFAULT = LLM_MODEL_GTP_4_O
LLM_MODEL_FALL_BACK = LLM_MODEL_GTP_3_5_TURBO

# Cloudflare R2 configs
CLOUDFLARE_R2_ACCESS_KEY = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY')
CLOUDFLARE_R2_SECRET_KEY = os.environ.get('CLOUDFLARE_R2_SECRET_KEY')
CLOUDFLARE_ACCOUNT_ID = os.environ.get('CLOUDFLARE_ACCOUNT_ID')

# Firebase configs
FIREBASE_SERVICE_ACCOUNT_KEY_LOCATION = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY_LOCATION')