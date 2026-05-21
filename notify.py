"""
notify.py
=========
Sends a warning/safe notification email directly to your Gmail inbox
for each scanned email — so you see the AI label as a real email message.

PHISHING email → sends you a red warning email
SAFE email     → sends you a green confirmation email (optional)

Uses Gmail API with send scope.
"""

import os
import json
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ── Gmail send scope (different from read-only!) ──────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE       = "token_send.json"   # separate token for send scope


# ── Auth ───────────────────────────────────────────────────────────────────────

def authenticate_send():
    """Authenticates with Gmail send scope."""
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ── Email HTML builders ────────────────────────────────────────────────────────

def build_phishing_html(email: dict) -> str:
    subject    = email.get("subject", "")
    sender     = email.get("sender", "")
    risk_score = email.get("risk_score", 0)
    explanation= email.get("explanation", "")
    reasons    = email.get("reasons", [])
    metadata   = email.get("metadata", {})
    sus_urls   = metadata.get("suspicious_urls", [])
    keywords   = metadata.get("keywords_found", [])

    reasons_html = "".join(f"<li>{r}</li>" for r in reasons[:5])
    urls_html    = "".join(
        f"<li style='color:#ff4444;font-family:monospace;font-size:12px;'>{u[:80]}</li>"
        for u in sus_urls[:3]
    ) or "<li>None</li>"
    kw_html = " ".join(
        f"<span style='background:#ff444422;color:#ff6666;border:1px solid #ff444444;"
        f"border-radius:4px;padding:2px 8px;font-size:12px;margin:2px;'>{k}</span>"
        for k in keywords[:8]
    ) or "None"

    return f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#0f1117;color:#eee;padding:24px;max-width:600px;margin:0 auto;">

  <!-- Header -->
  <div style="background:#ff4b4b;border-radius:10px 10px 0 0;padding:20px 24px;">
    <h1 style="margin:0;color:#fff;font-size:22px;">⚠️ Phishing Warning</h1>
    <p style="margin:6px 0 0;color:#ffcccc;font-size:13px;">
      AI Email Safety Scanner detected a suspicious email
    </p>
  </div>

  <!-- Body -->
  <div style="background:#1a1d27;border-radius:0 0 10px 10px;padding:24px;border:1px solid #ff4b4b44;">

    <!-- Original email info -->
    <div style="background:#12151f;border-radius:8px;padding:14px 16px;margin-bottom:16px;">
      <p style="margin:0 0 6px;font-size:12px;color:#888;">SUSPICIOUS EMAIL DETAILS</p>
      <p style="margin:0 0 4px;font-size:14px;"><b>Subject:</b> {subject}</p>
      <p style="margin:0 0 4px;font-size:14px;"><b>From:</b> {sender}</p>
      <p style="margin:0;font-size:14px;"><b>Risk Score:</b>
        <span style="color:#ff4b4b;font-weight:700;">{risk_score}/100</span>
      </p>
    </div>

    <!-- Risk bar -->
    <div style="background:#2a2d3a;border-radius:99px;height:8px;margin-bottom:16px;">
      <div style="background:#ff4b4b;width:{risk_score}%;height:8px;border-radius:99px;"></div>
    </div>

    <!-- AI Explanation -->
    <div style="border-left:3px solid #7c4dff;padding:12px 16px;background:#12151f;
                border-radius:0 8px 8px 0;margin-bottom:16px;">
      <p style="margin:0 0 4px;font-size:11px;color:#888;">AI EXPLANATION</p>
      <p style="margin:0;font-size:14px;color:#ccc;line-height:1.6;">{explanation}</p>
    </div>

    <!-- Evidence -->
    <div style="margin-bottom:16px;">
      <p style="font-size:12px;color:#888;margin:0 0 8px;">🚩 EVIDENCE FOUND</p>
      <ul style="margin:0;padding-left:20px;font-size:13px;color:#ffaaaa;line-height:1.8;">
        {reasons_html}
      </ul>
    </div>

    <!-- Suspicious URLs -->
    <div style="margin-bottom:16px;">
      <p style="font-size:12px;color:#888;margin:0 0 8px;">🔗 SUSPICIOUS URLS</p>
      <ul style="margin:0;padding-left:20px;line-height:1.8;">
        {urls_html}
      </ul>
    </div>

    <!-- Keywords -->
    <div style="margin-bottom:20px;">
      <p style="font-size:12px;color:#888;margin:0 0 8px;">🔑 PHISHING KEYWORDS DETECTED</p>
      <div>{kw_html}</div>
    </div>

    <!-- Warning box -->
    <div style="background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:8px;
                padding:14px 16px;text-align:center;">
      <p style="margin:0;font-size:14px;color:#ff6b6b;font-weight:600;">
        ⛔ Do NOT click any links or provide personal information in this email.
      </p>
    </div>

  </div>

  <p style="font-size:11px;color:#444;text-align:center;margin-top:16px;">
    Sent by AI Email Safety Scanner · Powered by TF-IDF + LangChain + FAISS
  </p>
</body>
</html>
"""


def build_safe_html(email: dict) -> str:
    subject    = email.get("subject", "")
    sender     = email.get("sender", "")
    risk_score = email.get("risk_score", 0)
    explanation= email.get("explanation", "")

    return f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#0f1117;color:#eee;padding:24px;max-width:600px;margin:0 auto;">

  <div style="background:#00c853;border-radius:10px 10px 0 0;padding:20px 24px;">
    <h1 style="margin:0;color:#fff;font-size:22px;">✅ Email Looks Safe</h1>
    <p style="margin:6px 0 0;color:#ccffdd;font-size:13px;">
      AI Email Safety Scanner — no phishing detected
    </p>
  </div>

  <div style="background:#1a1d27;border-radius:0 0 10px 10px;padding:24px;border:1px solid #00c85344;">

    <div style="background:#12151f;border-radius:8px;padding:14px 16px;margin-bottom:16px;">
      <p style="margin:0 0 6px;font-size:12px;color:#888;">EMAIL DETAILS</p>
      <p style="margin:0 0 4px;font-size:14px;"><b>Subject:</b> {subject}</p>
      <p style="margin:0 0 4px;font-size:14px;"><b>From:</b> {sender}</p>
      <p style="margin:0;font-size:14px;"><b>Risk Score:</b>
        <span style="color:#00c853;font-weight:700;">{risk_score}/100</span>
      </p>
    </div>

    <div style="background:#2a2d3a;border-radius:99px;height:8px;margin-bottom:16px;">
      <div style="background:#00c853;width:{risk_score}%;height:8px;border-radius:99px;"></div>
    </div>

    <div style="border-left:3px solid #00c853;padding:12px 16px;background:#12151f;
                border-radius:0 8px 8px 0;margin-bottom:16px;">
      <p style="margin:0 0 4px;font-size:11px;color:#888;">AI EXPLANATION</p>
      <p style="margin:0;font-size:14px;color:#ccc;line-height:1.6;">{explanation}</p>
    </div>

    <div style="background:#00c85318;border:1px solid #00c85344;border-radius:8px;
                padding:14px 16px;text-align:center;">
      <p style="margin:0;font-size:14px;color:#00e676;font-weight:600;">
        ✓ This email appears safe. Stay cautious with unexpected attachments.
      </p>
    </div>

  </div>

  <p style="font-size:11px;color:#444;text-align:center;margin-top:16px;">
    Sent by AI Email Safety Scanner · Powered by TF-IDF + LangChain + FAISS
  </p>
</body>
</html>
"""


# ── Send email ─────────────────────────────────────────────────────────────────

def send_notification(service, to_email: str, email: dict) -> bool:
    """
    Sends a notification email to your inbox about a scanned email.

    Args:
        service:    Authenticated Gmail service with send scope.
        to_email:   Your Gmail address to send the notification to.
        email:      Standardized email result dict from output.py

    Returns:
        True if sent successfully, False otherwise.
    """
    classification = email.get("classification", "UNKNOWN")
    subject        = email.get("subject", "")

    if classification == "PHISHING":
        notify_subject = f"⚠️ PHISHING DETECTED: {subject[:50]}"
        html_body      = build_phishing_html(email)
    else:
        notify_subject = f"✅ SAFE: {subject[:55]}"
        html_body      = build_safe_html(email)

    # Build MIME message
    msg = MIMEMultipart("alternative")
    msg["To"]      = to_email
    msg["From"]    = to_email
    msg["Subject"] = notify_subject

    # Attach HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Encode to base64url for Gmail API
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    try:
        service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()
        return True
    except Exception as e:
        print(f"  [Error] Failed to send: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def notify_all(
    results: list[dict],
    your_email: str,
    phishing_only: bool = True
) -> None:
    """
    Sends notification emails for all scanned emails.

    Args:
        results:      List of standardized output dicts from output.py
        your_email:   Your Gmail address
        phishing_only: If True, only sends warnings for PHISHING emails.
                       If False, sends notifications for ALL emails.
    """
    print(f"\n[Notify] Authenticating with Gmail send scope...")
    service = authenticate_send()

    to_send = results if not phishing_only else [
        r for r in results if r["classification"] == "PHISHING"
    ]

    if not to_send:
        print("[Notify] No emails to notify about.")
        return

    print(f"[Notify] Sending {len(to_send)} notification(s) to {your_email}...")

    sent  = 0
    failed = 0
    for email in to_send:
        subj = email.get("subject", "")[:50]
        cls  = email.get("classification", "?")
        print(f"  Sending [{cls}]: {subj}...")
        ok = send_notification(service, your_email, email)
        if ok:
            print(f"  ✓ Sent")
            sent += 1
        else:
            failed += 1

    print(f"\n[Done] {sent} notifications sent, {failed} failed.")
    print(f"[Done] Check your Gmail inbox — you should see the warning emails!")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AI Email Safety Scanner — Notify via Gmail")
    print("=" * 55)

    # Load standardized results
    try:
        with open("output.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("[Error] output.json not found. Run the full pipeline first.")
        exit(1)

    # ── SET YOUR EMAIL HERE ────────────────────────────────────────────────────
    YOUR_EMAIL = "vijayalaxmigona64@gmail.com"   # ← change this to your Gmail address
    # ──────────────────────────────────────────────────────────────────────────

    if YOUR_EMAIL == "your.email@gmail.com":
        print("\n[Setup] Open notify.py and set YOUR_EMAIL to your Gmail address.")
        print("  Example: YOUR_EMAIL = 'your.email@gmail.com'")
        exit(1)

    # Send only phishing warnings (change to False to send for all emails)
    notify_all(results, YOUR_EMAIL, phishing_only=True)
