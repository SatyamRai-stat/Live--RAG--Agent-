from src.tools.gmail_tool import get_gmail_service

service = get_gmail_service()

result = service.users().messages().list(userId="me", maxResults=5).execute()

for m in result.get("messages", []):
    msg = service.users().messages().get(
        userId="me",
        id=m["id"],
        format="metadata",
        metadataHeaders=["From", "Subject"],
    ).execute()
    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    print(headers.get("From"), "|", headers.get("Subject"))