"""
preprocess.py
=============
Cleans and structures raw email data fetched by gmail_fetcher.py.

What this file does:
  1. Removes HTML tags from email body
  2. Cleans invisible/junk characters (like those ͏ ͏ ͏ in snippets)
  3. Extracts all URLs using regex
  4. Extracts sender domain (e.g. linkedin.com from jobalerts-noreply@linkedin.com)
  5. Extracts phishing-relevant keywords found in the text
  6. Returns a clean structured dict ready for detector.py

Dependencies:
  beautifulsoup4, lxml (already in requirements.txt)
"""

import re
import json
from urllib.parse import urlparse
from bs4 import BeautifulSoup


# ── Phishing keyword list (used in keyword extraction) ────────────────────────

PHISHING_KEYWORDS = [
    # Urgency triggers
    "urgent", "immediately", "action required", "act now", "limited time",
    "expires", "expiring", "deadline", "last chance", "final notice",
    # Account / login bait
    "verify", "verify your account", "confirm your account", "validate",
    "login", "log in", "sign in", "your account", "account suspended",
    "account locked", "account disabled", "unusual activity", "suspicious activity",
    # Credential harvesting
    "password", "reset password", "update your password", "username",
    "enter your details", "confirm your details", "update your information",
    # Financial lures
    "bank", "banking", "credit card", "debit card", "payment", "invoice",
    "refund", "transaction", "wire transfer", "prize", "winner", "won",
    "lottery", "claim your", "free gift", "reward",
    # Fear / threat
    "your account will be", "will be terminated", "will be suspended",
    "will be closed", "legal action", "law enforcement", "arrest",
    # Click bait
    "click here", "click the link", "click below", "follow this link",
    "open the attachment", "download the file",
    # Generic phishing phrases
    "dear customer", "dear user", "dear account holder",
    "we have noticed", "we detected", "kindly", "do not ignore",
]


# ── HTML Cleaning ──────────────────────────────────────────────────────────────

def strip_html(html_text: str) -> str:
    """
    Removes all HTML tags and returns plain text.

    Uses BeautifulSoup with the lxml parser for robust HTML handling.
    Falls back gracefully if the input isn't HTML at all.

    Args:
        html_text: Raw email body (may contain HTML tags).

    Returns:
        Plain text with all tags removed.
    """
    if not html_text:
        return ""

    try:
        soup = BeautifulSoup(html_text, "lxml")

        # Remove script and style blocks entirely (not just their tags)
        for tag in soup(["script", "style", "head", "meta", "link"]):
            tag.decompose()

        # Get text, using space as separator between block elements
        text = soup.get_text(separator=" ")
    except Exception:
        # If parsing fails, fall back to simple regex tag removal
        text = re.sub(r"<[^>]+>", " ", html_text)

    return text


def clean_text(raw_text: str) -> str:
    """
    Cleans raw text by removing invisible characters, extra whitespace,
    and non-printable junk (like the ͏ characters in LinkedIn snippets).

    Args:
        raw_text: Text to clean (plain text or already stripped HTML).

    Returns:
        Clean, normalized text string.
    """
    if not raw_text:
        return ""

    # Remove zero-width and invisible Unicode characters
    # U+034F (combining grapheme joiner), U+200B (zero-width space), etc.
    text = re.sub(r"[\u034f\u200b\u200c\u200d\u00ad\ufeff]", "", raw_text)

    # Remove other non-printable control characters (except newline/tab)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── URL Extraction ─────────────────────────────────────────────────────────────

# Regex that matches http/https URLs (including those with complex paths/params)
URL_REGEX = re.compile(
    r"https?://"                          # scheme
    r"(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}" # domain
    r"(?:/[^\s<>\"']*)?",                 # optional path
    re.IGNORECASE
)

def extract_urls(text: str) -> list[str]:
    """
    Finds all URLs in the given text using regex.

    Args:
        text: Plain text (HTML already stripped).

    Returns:
        List of unique URL strings found in the text.
    """
    urls = URL_REGEX.findall(text)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def is_suspicious_url(url: str) -> bool:
    """
    Checks a single URL for common phishing red flags.

    Red flags:
      - Uses an IP address instead of a domain name
      - Has an unusually long domain (>40 chars)
      - Contains suspicious TLDs often abused by phishers
      - Contains multiple subdomains (e.g. secure.paypal.com.evil.com)
      - Contains URL shorteners (bit.ly, tinyurl, etc.)
      - Contains suspicious keywords in the URL path

    Args:
        url: A URL string to check.

    Returns:
        True if the URL looks suspicious, False otherwise.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path   = parsed.path.lower()
    except Exception:
        return False

    # Red flag 1: IP address instead of domain
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}", domain):
        return True

    # Red flag 2: Very long domain (legitimate domains are short)
    if len(domain) > 40:
        return True

    # Red flag 3: Suspicious TLDs commonly used in phishing
    suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
                       ".click", ".download", ".loan", ".work", ".racing"]
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        return True

    # Red flag 4: URL shorteners (hide the real destination)
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
                  "short.link", "cutt.ly", "rebrand.ly"]
    if any(s in domain for s in shorteners):
        return True

    # Red flag 5: Suspicious keywords in URL path
    suspicious_path_keywords = ["login", "verify", "account", "secure",
                                 "update", "confirm", "password", "banking"]
    if any(kw in path for kw in suspicious_path_keywords):
        return True

    # Red flag 6: Many subdomains (e.g. secure.login.paypal.com.evil.com)
    subdomain_count = domain.count(".")
    if subdomain_count > 3:
        return True

    return False


# ── Sender Domain Extraction ───────────────────────────────────────────────────

def extract_sender_domain(sender: str) -> str:
    """
    Extracts the domain from a sender string.

    Handles formats like:
      - "LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>"  → "linkedin.com"
      - "noreply@paypal.com"                                    → "paypal.com"
      - "John Doe"                                              → ""

    Args:
        sender: The raw 'From' header value.

    Returns:
        Domain string (lowercase), or empty string if not found.
    """
    # Look for email address pattern inside angle brackets first
    match = re.search(r"<([^>]+)>", sender)
    email = match.group(1) if match else sender.strip()

    # Extract domain from email address
    if "@" in email:
        domain = email.split("@")[-1].strip().lower()
        # Remove any trailing > or whitespace
        domain = re.sub(r"[>\s]", "", domain)
        return domain

    return ""


def check_domain_mismatch(sender_domain: str, urls: list[str]) -> bool:
    """
    Checks if links in the email point to a domain different from the sender.

    This is a classic phishing signal — an email claims to be from paypal.com
    but the links go to evil-site.com.

    Args:
        sender_domain: Domain extracted from the From header.
        urls:          List of URLs found in the email body.

    Returns:
        True if a mismatch is detected, False otherwise.
    """
    if not sender_domain or not urls:
        return False

    # Known legitimate redirect/tracking domains to ignore
    # (e.g. email.linkedin.com is LinkedIn's email tracking domain)
    allowed_redirects = [
        "unsubscribe", "email.", "mail.", "click.", "track.", "links.",
        "mailchimp", "sendgrid", "constantcontact", "klaviyo",
        "google.com", "goo.gl", "youtube.com",
    ]

    mismatches = 0
    for url in urls:
        try:
            link_domain = urlparse(url).netloc.lower()
        except Exception:
            continue

        # Skip if it's the same domain or a subdomain of sender
        if sender_domain in link_domain:
            continue

        # Skip known email marketing/redirect services
        if any(r in link_domain for r in allowed_redirects):
            continue

        mismatches += 1

    # Flag if more than half the links point to foreign domains
    return mismatches > len(urls) / 2


# ── Keyword Extraction ─────────────────────────────────────────────────────────

def extract_keywords(text: str) -> list[str]:
    """
    Finds phishing-related keywords present in the email text.

    Args:
        text: Cleaned plain text of the email.

    Returns:
        List of matched phishing keywords (lowercase).
    """
    text_lower = text.lower()
    found = []
    for keyword in PHISHING_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return found


# ── Main Preprocessing Function ────────────────────────────────────────────────

def preprocess_email(email: dict) -> dict:
    """
    Takes a raw email dict (from gmail_fetcher.py) and returns a clean,
    structured dict ready for the ML detector and AI agent.

    Input email dict keys:
        id, subject, sender, date, snippet, body

    Output (preprocessed) dict keys:
        id, subject, sender, date,
        clean_body     - HTML stripped + cleaned body text
        clean_snippet  - cleaned snippet
        full_text      - subject + body combined (for ML)
        urls           - list of URLs found
        suspicious_urls- list of URLs flagged as suspicious
        sender_domain  - domain from the From header
        domain_mismatch- True if links don't match sender domain
        keywords       - list of phishing keywords found
        keyword_count  - number of phishing keywords
        url_count      - total URLs found
        suspicious_url_count - number of suspicious URLs

    Args:
        email: Raw email dict from gmail_fetcher.py

    Returns:
        Preprocessed email dict with all extracted features.
    """
    # ── 1. Clean body ──────────────────────────────────────────────────────────
    raw_body    = email.get("body", "") or ""
    stripped    = strip_html(raw_body)           # Remove HTML tags
    clean_body  = clean_text(stripped)           # Remove junk characters

    # ── 2. Clean snippet ──────────────────────────────────────────────────────
    raw_snippet   = email.get("snippet", "") or ""
    clean_snippet = clean_text(raw_snippet)

    # ── 3. Clean subject ──────────────────────────────────────────────────────
    subject = clean_text(email.get("subject", "") or "")

    # ── 4. Combine subject + body for ML model ────────────────────────────────
    # The ML model in Phase 3 works on a single text string
    full_text = f"{subject} {clean_body}".strip()

    # ── 5. Extract URLs (from raw body to catch hidden links in HTML) ─────────
    # Search both raw HTML body and cleaned text to catch all URLs
    urls_from_html = extract_urls(raw_body)
    urls_from_text = extract_urls(clean_body)
    # Merge and deduplicate
    all_urls = list(dict.fromkeys(urls_from_html + urls_from_text))

    # ── 6. Flag suspicious URLs ───────────────────────────────────────────────
    suspicious_urls = [url for url in all_urls if is_suspicious_url(url)]

    # ── 7. Extract sender domain ──────────────────────────────────────────────
    sender        = email.get("sender", "") or ""
    sender_domain = extract_sender_domain(sender)

    # ── 8. Check domain mismatch ──────────────────────────────────────────────
    domain_mismatch = check_domain_mismatch(sender_domain, all_urls)

    # ── 9. Extract phishing keywords ──────────────────────────────────────────
    keywords = extract_keywords(full_text)

    # ── 10. Assemble final dict ───────────────────────────────────────────────
    return {
        # Identity
        "id":                   email.get("id", ""),
        "subject":              subject,
        "sender":               sender,
        "date":                 email.get("date", ""),

        # Cleaned text
        "clean_body":           clean_body,
        "clean_snippet":        clean_snippet,
        "full_text":            full_text,

        # URL features
        "urls":                 all_urls,
        "suspicious_urls":      suspicious_urls,
        "url_count":            len(all_urls),
        "suspicious_url_count": len(suspicious_urls),

        # Domain features
        "sender_domain":        sender_domain,
        "domain_mismatch":      domain_mismatch,

        # Keyword features
        "keywords":             keywords,
        "keyword_count":        len(keywords),
    }


def preprocess_all(emails: list[dict]) -> list[dict]:
    """
    Preprocesses a list of raw email dicts.

    Args:
        emails: List of raw email dicts from gmail_fetcher.py

    Returns:
        List of preprocessed email dicts.
    """
    processed = []
    for i, email in enumerate(emails):
        print(f"[Preprocess] Processing email {i + 1}/{len(emails)}: {email.get('subject', '')[:60]}")
        result = preprocess_email(email)
        processed.append(result)
    print(f"\n[Done] Preprocessed {len(processed)} emails.")
    return processed


# ── Save helper ────────────────────────────────────────────────────────────────

def save_preprocessed(data: list[dict], path: str = "preprocessed.json") -> None:
    """Saves preprocessed email list to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Save] Preprocessed data saved to '{path}'.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load emails fetched in Phase 1
    try:
        with open("emails.json", "r", encoding="utf-8") as f:
            emails = json.load(f)
    except FileNotFoundError:
        print("[Error] emails.json not found. Run gmail_fetcher.py first.")
        exit(1)

    # Preprocess all emails
    processed = preprocess_all(emails)

    # Print a sample result
    if processed:
        sample = processed[0]
        print("\n── Sample Preprocessed Email ─────────────────────────────")
        print(f"  Subject        : {sample['subject']}")
        print(f"  Sender Domain  : {sample['sender_domain']}")
        print(f"  Clean Body     : {sample['clean_body'][:200]}...")
        print(f"  URLs Found     : {sample['url_count']}")
        print(f"  Suspicious URLs: {sample['suspicious_url_count']}")
        print(f"  Domain Mismatch: {sample['domain_mismatch']}")
        print(f"  Keywords Found : {sample['keywords']}")
        print(f"  Keyword Count  : {sample['keyword_count']}")

    # Save for Phase 3
    save_preprocessed(processed)
