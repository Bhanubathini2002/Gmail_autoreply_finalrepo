# read_gmail_to_milvus.py
"""
Reads emails from Gmail API → generates embeddings via Ollama → stores in Milvus.
Requires:
- credentials.json (from Google Cloud)
- token.json (generated after OAuth login)
- Ollama running (e.g. ollama serve)
- Milvus running in Docker
"""

import base64
import re
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle, os.path

from embedder import OllamaEmbedder
from vector_store import GmailVectorStore


# ============================================================
# Gmail Setup
# ============================================================

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_gmail_service():
    """Authenticate and return Gmail API service"""
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)


def clean_text(text):
    """Remove HTML tags, newlines, and excess spaces"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def read_emails(max_results=5):# increase the capacity
    """Fetch latest emails"""
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")

        # Extract message body
        body = ""
        if 'data' in msg_data['payload']['body']:
            body = base64.urlsafe_b64decode(msg_data['payload']['body']['data']).decode('utf-8', errors='ignore')
        elif 'parts' in msg_data['payload']:
            for part in msg_data['payload']['parts']:
                if 'data' in part['body']:
                    body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

        emails.append({
            "subject": subject,
            "from_email": sender,
            "body": clean_text(body)
        })
    return emails


# ============================================================
# Main Pipeline
# ============================================================

if __name__ == "__main__":
    print("📩 Fetching Gmail messages...")
    emails = read_emails(max_results=5)
    print(f"✅ Retrieved {len(emails)} emails.")

    embedder = OllamaEmbedder()
    store = GmailVectorStore(dim=768)  # 768 is embedding size for nomic-embed-text

    for email in emails:
        text_content = f"Subject: {email['subject']}\nFrom: {email['from_email']}\nBody: {email['body']}"
        embedding = embedder.embed(text_content)
        store.insert_email(email['subject'], email['from_email'], email['body'], embedding)

    print("✅ All Gmail emails embedded and stored in Milvus.")





'''# for sync of gmails instead of indexing 
SYNC_FILE = "last_synced.json"

def get_last_synced_id():
    """Load last synced Gmail message ID from file"""
    import json, os
    if not os.path.exists(SYNC_FILE):
        return None
    with open(SYNC_FILE, "r") as f:
        data = json.load(f)
    return data.get("last_id")

def update_last_synced_id(msg_id):
    """Update the tracker with the latest Gmail message ID"""
    import json
    with open(SYNC_FILE, "w") as f:
        json.dump({"last_id": msg_id}, f)

        

def read_emails(max_results=100):
    """Fetch only new Gmail messages since last sync"""
    service = get_gmail_service()
    last_synced = get_last_synced_id()
    print(f"🔍 Last synced ID: {last_synced}")

    query_params = {
        "userId": "me",
        "maxResults": max_results,
    }

    # If last synced message exists, use Gmail 'q' search with newer_than
    if last_synced:
        query_params["q"] = f"newer_than:1d"  # or adjust timeframe
        # alternatively, you can filter manually after fetching

    response = service.users().messages().list(**query_params).execute()
    messages = response.get("messages", [])
    emails = []

    for msg in messages:
        msg_id = msg["id"]
        msg_data = service.users().messages().get(userId="me", id=msg_id).execute()

        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")

        body = ""
        if "data" in msg_data["payload"]["body"]:
            body = base64.urlsafe_b64decode(msg_data["payload"]["body"]["data"]).decode("utf-8", errors="ignore")
        elif "parts" in msg_data["payload"]:
            for part in msg_data["payload"]["parts"]:
                if "data" in part["body"]:
                    body += base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")

        emails.append({
            "id": msg_id,
            "subject": subject,
            "from_email": sender,
            "body": clean_text(body)
        })

    # Save newest message ID (the first one returned is latest)
    if emails:
        update_last_synced_id(emails[0]["id"])
        print(f"✅ Updated last synced ID: {emails[0]['id']}")

    return emails
'''