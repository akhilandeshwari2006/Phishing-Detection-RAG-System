# SETUP.md — Google Cloud & Gmail API Setup Guide
# ================================================
# Follow every step in order. Takes ~10 minutes.

## Step 1 — Create a Google Cloud Project

1. Go to: https://console.cloud.google.com/
2. Click the project dropdown (top-left) → "New Project"
3. Name it: email-safety-scanner
4. Click "Create"

## Step 2 — Enable the Gmail API

1. In the Cloud Console, go to:
   APIs & Services → Library
2. Search for "Gmail API"
3. Click it → click "Enable"

## Step 3 — Configure the OAuth Consent Screen

1. Go to: APIs & Services → OAuth consent screen
2. Choose "External" → click "Create"
3. Fill in:
   - App name: Email Safety Scanner
   - User support email: (your Gmail)
   - Developer contact: (your Gmail)
4. Click "Save and Continue" through all steps
5. On the "Test users" page → click "Add Users" → add your Gmail address
6. Click "Save and Continue" → "Back to Dashboard"

## Step 4 — Create OAuth 2.0 Credentials

1. Go to: APIs & Services → Credentials
2. Click "+ Create Credentials" → "OAuth client ID"
3. Application type: "Desktop app"
4. Name: email-scanner-client
5. Click "Create"
6. Click "Download JSON"
7. Rename the downloaded file to: credentials.json
8. Place it in your project root:
   email_scanner/
   └── credentials.json   ← here

## Step 5 — Install Python Dependencies

  # Create a virtual environment (recommended)
  python -m venv venv

  # Activate it:
  # macOS/Linux:
  source venv/bin/activate
  # Windows:
  venv\Scripts\activate

  # Install all packages:
  pip install -r requirements.txt

## Step 6 — Run the Fetcher

  python gmail_fetcher.py

  What happens:
  1. A browser window opens → log in with your Google account
  2. Grant permission to "read your email"
  3. token.json is saved (you won't need to log in again)
  4. 10 emails are fetched and saved to emails.json

## File Roles

  credentials.json  → Google OAuth client secret (never commit to git!)
  token.json        → Your access token (auto-created, never commit to git!)
  emails.json       → Fetched emails (used by all later phases)

## .gitignore (add this to avoid leaking secrets)

  credentials.json
  token.json
  .env
  __pycache__/
  venv/

## Troubleshooting

  Error: "credentials.json not found"
  → Make sure you downloaded and renamed the file correctly (Step 4).

  Error: "Access blocked: app not verified"
  → You're in test mode. Click "Advanced" → "Go to email-scanner (unsafe)".
     This is normal for personal/dev projects not submitted for Google verification.

  Error: "Token has been expired or revoked"
  → Delete token.json and re-run. A fresh browser login will fix it.

  Error: "HttpError 403"
  → Gmail API may not be enabled. Double-check Step 2.
