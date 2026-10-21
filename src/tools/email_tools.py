import base64
from langchain_core.tools import tool
from src.tools.gmail_tool import get_gmail_service
import re

def get_header(headers, name):
    for h in headers:
        if h["name"] == name:
            return h["value"]
    return ""

#used for decoding data 
def get_text(payload):
    # emails can have several parts; look for the plain-text one
    if payload.get("mimeType") == "text/plain" and payload["body"].get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    for part in payload.get("parts", []):
        text = get_text(part)
        if text:
            return text
    return ""


@tool
def search_emails(query: str, max_results: int = 5) -> list:
    """Search Gmail using Gmail search syntax, for example
    'from:linkedin newer_than:7d' or 'subject:interview'.
    Returns id, sender, subject, date and a short snippet for each match."""
    service = get_gmail_service()
    max_results = min(max_results, 10)  # safety cap

    result = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()

    emails = []
    for m in result.get("messages", []):
        msg = service.users().messages().get(
            userId="me",
            id=m["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = msg["payload"]["headers"]
        emails.append({
            "id": m["id"],
            "from": get_header(headers, "From"),
            "subject": get_header(headers, "Subject"),
            "date": get_header(headers, "Date"),
            "snippet": msg.get("snippet", ""),
        })
    return emails


@tool
def read_email(email_id: str) -> str:
    """Read the plain-text body of one email. Use an id returned by search_emails."""
    service = get_gmail_service()
    msg = service.users().messages().get(
        userId="me", id=email_id, format="full"
    ).execute()

    body = get_text(msg["payload"])  # 1. get the text (this is the line you deleted)

    if not body:                     # 2. check for empty first
        return "(no plain-text body found)"

    body = re.sub(r"https?://\S+", "[link]", body)  # 3. remove long tracking URLs
    body = re.sub(r"\s+", " ", body)                # 4. collapse blank space
    return body[:1500]                              # 5. cut to save tokens