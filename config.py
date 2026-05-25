import os
from dotenv import load_dotenv

#1. Carica il file .env
load_dotenv()
#2. Esporta le chiavi come costanti pulite
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")