from src.tools.email_tools import search_emails, read_email

emails = search_emails.invoke({"query": "from:linkedin", "max_results": 3})

for e in emails:
    print(e["id"], "|", e["from"], "|", e["subject"])

if emails:
    print("\n--- First email body (first 500 chars) ---")
    print(read_email.invoke({"email_id": emails[0]["id"]})[:500])