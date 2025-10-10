# smart_reply.py
"""
Smart Gmail Responder:
Fetches a new Gmail message, retrieves similar emails from Milvus,
and drafts a contextual reply using Ollama.
"""

import base64
import re
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle, os

from embedder import OllamaEmbedder
from vector_store import GmailVectorStore


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# ============================================================
# Gmail setup
# ============================================================
def get_gmail_service():
    """Authenticate and return Gmail API service (JSON token format)"""
    from google.oauth2.credentials import Credentials
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_latest_email():
    """Fetch the most recent email"""
    service = get_gmail_service()
    result = service.users().messages().list(userId='me', maxResults=1).execute()
    message_id = result['messages'][0]['id']
    msg = service.users().messages().get(userId='me', id=message_id).execute()

    headers = msg['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
    sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")

    body = ""
    if 'data' in msg['payload']['body']:
        body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8', errors='ignore')
    elif 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if 'data' in part['body']:
                body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

    body = clean_text(body)
    return {
        "id": message_id,
        "subject": subject,
        "from_email": sender,
        "body": body
    }


# ============================================================
# Reply generation logic
# ============================================================

def get_similar_context(email_text, top_k=3):
    """Retrieve similar emails from Milvus"""
    embedder = OllamaEmbedder()
    store = GmailVectorStore(dim=768)
    qvec = embedder.embed(email_text)
    hits = store.search_similar(qvec, limit=top_k)

    context_blocks = []
    for hit in hits:
        subj = hit.entity.get("subject")
        sender = hit.entity.get("from_email")
        body = hit.entity.get("body")
        context_blocks.append(f"Subject: {subj}\nFrom: {sender}\nBody: {body}\n---")

    return "\n".join(context_blocks)

def generate_reply_with_ollama(email_text, similar_context):
    """Generate a smart reply using Ollama LLM"""
    import json
    import requests

    payload = {
        "model": "mistral:instruct",   # or whichever model you have pulled {{{{{llama3}}}}}   
        "prompt": f"""
You are a helpful and professional email assistant.

Use the previous emails below as reference to match tone and relevance.

CONTEXT:
{similar_context}

NEW EMAIL:
{email_text}

Write a short, polite, human-like reply for this email.
""",
        "stream": True
    }

    reply_text = ""
    try:
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            chunk = data["response"]
                            reply_text += chunk
                            print(chunk, end="", flush=True)  # live stream to console
                    except json.JSONDecodeError:
                        continue
        print("\n")
    except Exception as e:
        print(f"❌ Error generating reply: {e}")

    if not reply_text.strip():
        reply_text = "(No reply generated — model returned empty response.)"

    return reply_text.strip()

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("📩 Fetching latest email...")
    latest_email = get_latest_email()
    email_text = f"Subject: {latest_email['subject']}\nFrom: {latest_email['from_email']}\nBody: {latest_email['body']}"
    print(f"✅ Got email: {latest_email['subject']} from {latest_email['from_email']}")

    print("\n🔍 Retrieving similar context from Milvus...")
    similar_context = get_similar_context(email_text)
    print(f"✅ Retrieved related context ({len(similar_context)} chars)")

    print("\n🧠 Generating reply using Ollama...")
    reply = generate_reply_with_ollama(email_text, similar_context)

    print("\n💬 Suggested Reply:\n")
    print(reply)
