"""
gmail_fetcher.py
================
Fetches emails from Gmail using the Gmail API with OAuth 2.0.

What this file does:
  1. Authenticates the user via Google OAuth 2.0 (browser pop-up, first run only)
  2. Saves the token so future runs don't need re-authentication
  3. Fetches the latest N emails from the inbox
  4. Returns each email as a structured dict:
       { id, subject, sender, snippet, body, date }

Dependencies (install via requirements.txt):
  google-auth, google-auth-oauthlib, google-auth-httplib2, google-api-python-client
"""

import os
import base64
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ── Configuration ─────────────────────────────────────────────────────────────

# Gmail API scope: read-only is safest for a scanner
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Paths (relative to project root)
CREDENTIALS_FILE = "credentials.json"   # Downloaded from Google Cloud Console
TOKEN_FILE       = "token.json"         # Auto-created after first login


# ── Authentication ─────────────────────────────────────────────────────────────

def authenticate() -> object:
    """
    Runs OAuth 2.0 flow and returns an authenticated Gmail API service.

    First run:
      - Opens a browser window for Google login + consent.
      - Saves the token to token.json for future runs.

    Subsequent runs:
      - Loads token.json directly (no browser needed).
      - Auto-refreshes the token if it has expired.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Gmail service object.
    """
    creds = None

    # Load existing token if it exists
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid credentials, trigger OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Silently refresh an expired token
            creds.refresh(Request())
        else:
            # First-time login: opens browser
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"'{CREDENTIALS_FILE}' not found.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials.\n"
                    "See SETUP.md for full instructions."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # Opens browser, picks a free port

        # Save token for next run
        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())
        print(f"[Auth] Token saved to '{TOKEN_FILE}'.")

    # Build and return the Gmail API service
    service = build("gmail", "v1", credentials=creds)
    print("[Auth] Authenticated successfully.")
    return service


# ── Email Fetching ─────────────────────────────────────────────────────────────

def get_message_ids(service, max_results: int = 20, label: str = "INBOX") -> list[str]:
    """
    Retrieves a list of message IDs from the specified Gmail label.

    Args:
        service:     Authenticated Gmail service object.
        max_results: Number of emails to fetch (default 20, max 500).
        label:       Gmail label to fetch from (default "INBOX").

    Returns:
        List of message ID strings.
    """
    try:
        response = (
            service.users()
            .messages()
            .list(userId="me", labelIds=[label], maxResults=max_results)
            .execute()
        )
        messages = response.get("messages", [])
        ids = [msg["id"] for msg in messages]
        print(f"[Fetch] Found {len(ids)} message IDs in {label}.")
        return ids

    except HttpError as e:
        print(f"[Error] Failed to list messages: {e}")
        return []


def decode_body(payload: dict) -> str:
    """
    Recursively decodes the email body from a Gmail message payload.

    Gmail stores email bodies in base64url encoding, sometimes nested in
    multipart MIME parts. This function handles both plain-text and HTML bodies.

    Args:
        payload: The 'payload' field from a Gmail message object.

    Returns:
        Decoded body text (plain-text preferred, falls back to HTML).
    """
    body_text = ""
    mime_type = payload.get("mimeType", "")

    if mime_type in ("text/plain", "text/html"):
        # Direct body — decode base64url
        data = payload.get("body", {}).get("data", "")
        if data:
            body_text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    elif "multipart" in mime_type:
        # Multipart email: iterate over parts, prefer plain text
        parts = payload.get("parts", [])
        plain_text = ""
        html_text  = ""

        for part in parts:
            part_mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data", "")
            if not data:
                # Recurse for nested multipart
                decoded = decode_body(part)
                if part_mime == "text/plain":
                    plain_text = decoded
                elif part_mime == "text/html":
                    html_text = decoded
            else:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if part_mime == "text/plain":
                    plain_text = decoded
                elif part_mime == "text/html":
                    html_text = decoded

        # Prefer plain text; fall back to HTML
        body_text = plain_text if plain_text else html_text

    return body_text


def parse_headers(headers: list[dict]) -> dict:
    """
    Extracts Subject, From, and Date from Gmail message headers.

    Args:
        headers: List of {'name': ..., 'value': ...} dicts from Gmail API.

    Returns:
        Dict with keys: subject, sender, date.
    """
    header_map = {h["name"].lower(): h["value"] for h in headers}
    return {
        "subject": header_map.get("subject", "(No Subject)"),
        "sender":  header_map.get("from",    "(Unknown Sender)"),
        "date":    header_map.get("date",    "(Unknown Date)"),
    }


def fetch_email(service, msg_id: str) -> dict:
    """
    Fetches and parses a single email by its Gmail message ID.

    Args:
        service: Authenticated Gmail service object.
        msg_id:  Gmail message ID string.

    Returns:
        Dict with keys: id, subject, sender, date, snippet, body.
        Returns None if the fetch fails.
    """
    try:
        # 'full' format returns headers + body; 'metadata' returns headers only
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        payload  = msg.get("payload", {})
        headers  = payload.get("headers", [])
        snippet  = msg.get("snippet", "")          # Short preview Gmail generates
        body     = decode_body(payload)

        parsed = parse_headers(headers)
        parsed.update({
            "id":      msg_id,
            "snippet": snippet,
            "body":    body,
        })
        return parsed

    except HttpError as e:
        print(f"[Error] Failed to fetch message {msg_id}: {e}")
        return None


def fetch_emails(max_results: int = 20, label: str = "INBOX") -> list[dict]:
    """
    Main entry point. Authenticates, fetches, and returns structured emails.

    Args:
        max_results: Number of emails to retrieve (default 20).
        label:       Gmail label to read from (default "INBOX").

    Returns:
        List of email dicts: [{ id, subject, sender, date, snippet, body }, ...]
    """
    service = authenticate()
    ids     = get_message_ids(service, max_results=max_results, label=label)

    emails = []
    for i, msg_id in enumerate(ids):
        print(f"[Fetch] Fetching email {i + 1}/{len(ids)} (id={msg_id})...")
        email = fetch_email(service, msg_id)
        if email:
            emails.append(email)

    print(f"\n[Done] Fetched {len(emails)} emails.")
    return emails


# ── Save to JSON (optional helper) ────────────────────────────────────────────

def save_emails_to_json(emails: list[dict], output_path: str = "emails.json") -> None:
    """
    Saves the fetched emails to a JSON file for offline testing/debugging.

    Args:
        emails:      List of email dicts returned by fetch_emails().
        output_path: Where to save the file (default 'emails.json').
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)
    print(f"[Save] Emails saved to '{output_path}'.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run directly to test the fetcher
    emails = fetch_emails(max_results=10)

    if emails:
        print("\n── Sample Email ──────────────────────────────")
        sample = emails[0]
        print(f"  Subject : {sample['subject']}")
        print(f"  From    : {sample['sender']}")
        print(f"  Date    : {sample['date']}")
        print(f"  Snippet : {sample['snippet'][:120]}...")
        print(f"  Body    : {sample['body'][:200]}...")

    # Save all emails to disk for use in later phases
    save_emails_to_json(emails)
