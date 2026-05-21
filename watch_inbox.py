"""
watch_inbox.py
==============
Watches your Gmail inbox and ONLY scans NEW emails as they arrive.
Runs continuously in the background.
Checks every 30 seconds — but only processes emails it hasn't seen before.

How it works:
  1. Remembers which email IDs it has already scanned (saved in seen_ids.json)
  2. Every 30 seconds, checks for new emails
  3. If a NEW email arrives → runs full pipeline on just that email
  4. Sends you a PHISHING warning or SAFE notification instantly
  5. Never re-scans old emails

Run with:
  python watch_inbox.py
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

# Import all pipeline modules
from gmail_fetcher   import authenticate, fetch_email, get_message_ids
from preprocess      import preprocess_email
from detector        import detect, get_model
from agent           import build_agent_chain, analyze_email
from rag             import get_rag
from llm             import generate_explanation, get_client
from output          import standardize
from notify          import authenticate_send, send_notification

# ── Config ─────────────────────────────────────────────────────────────────────
CHECK_INTERVAL = 30        # seconds between inbox checks
SEEN_IDS_FILE  = "seen_ids.json"
YOUR_EMAIL     = "vijayalaxmigona64@gmail.com"   # ← change this to your Gmail


# ── Seen IDs tracker ───────────────────────────────────────────────────────────

def load_seen_ids() -> set:
    """Loads the set of already-scanned email IDs from disk."""
    if Path(SEEN_IDS_FILE).exists():
        with open(SEEN_IDS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set) -> None:
    """Saves the set of scanned email IDs to disk."""
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(list(seen_ids), f)


# ── Process a single new email ─────────────────────────────────────────────────

def process_new_email(
    gmail_service,
    notify_service,
    msg_id: str,
    agent_chain,
    rag,
    llm_client,
) -> None:
    """
    Runs the full pipeline on a single new email and sends notification.

    Args:
        gmail_service:  Authenticated Gmail read service
        notify_service: Authenticated Gmail send service
        msg_id:         Gmail message ID of the new email
        agent_chain:    LangChain agent chain
        rag:            PhishingRAG instance
        llm_client:     OpenAI client (or None for fallback)
    """
    print(f"\n[New Email] Processing {msg_id}...")

    # Step 1: Fetch
    raw_email = fetch_email(gmail_service, msg_id)
    if not raw_email:
        print(f"  [Skip] Could not fetch email {msg_id}")
        return

    print(f"  Subject : {raw_email.get('subject','')[:60]}")
    print(f"  From    : {raw_email.get('sender','')[:50]}")

    # Step 2: Preprocess
    preprocessed = preprocess_email(raw_email)

    # Step 3: Detect (ML + rules)
    detection = detect(preprocessed)
    merged = {**preprocessed, **detection}

    # Step 4: Agent reasoning
    result = analyze_email(merged, agent_chain)

    # Step 5: RAG enrichment
    result = rag.enrich_email(result, top_k=3)

    # Simplify RAG matches for JSON
    rag_matches = result.get("rag_matches", [])
    result["rag_top_matches"] = [
        {
            "subject":     m["example"]["subject"],
            "attack_type": m["example"]["attack_type"],
            "similarity":  m["similarity"],
            "risk_score":  m["example"]["risk_score"],
        }
        for m in rag_matches
    ]

    # Step 6: LLM explanation
    explanation = generate_explanation(result, llm_client)
    result["llm_explanation"] = explanation

    # Step 7: Standardize
    final = standardize(result)

    # Step 8: Send notification
    verdict    = final.get("classification", "UNKNOWN")
    risk_score = final.get("risk_score", 0)

    print(f"  Verdict : {verdict} (risk: {risk_score}/100)")
    print(f"  Explain : {explanation[:80]}...")

    send_notification(notify_service, YOUR_EMAIL, final)
    print(f"  ✓ Notification sent to {YOUR_EMAIL}")


# ── Main watcher loop ──────────────────────────────────────────────────────────

def watch():
    """
    Main loop — checks for new emails every CHECK_INTERVAL seconds.
    Only processes emails that haven't been seen before.
    """
    if YOUR_EMAIL == "your.email@gmail.com":
        print("[Setup] Open watch_inbox.py and set YOUR_EMAIL to your Gmail address.")
        print("  Example: YOUR_EMAIL = 'vijay@gmail.com'")
        return

    print("=" * 55)
    print("  AI Email Safety Scanner — Live Inbox Watcher")
    print("=" * 55)
    print(f"  Watching: {YOUR_EMAIL}")
    print(f"  Check interval: every {CHECK_INTERVAL} seconds")
    print(f"  Press Ctrl+C to stop")
    print("=" * 55)

    # ── Initialize all components once (not per email) ─────────────────────────
    print("\n[Init] Loading all pipeline components...")

    print("  → Authenticating Gmail (read)...")
    gmail_service = authenticate()

    print("  → Authenticating Gmail (send)...")
    notify_service = authenticate_send()

    print("  → Loading ML model...")
    get_model()   # loads/trains model into memory

    print("  → Building LangChain agent...")
    try:
        agent_chain = build_agent_chain()
    except Exception as e:
        print(f"  → Agent unavailable ({e}), using fallback.")
        agent_chain = None

    print("  → Loading RAG index...")
    rag = get_rag()

    print("  → Connecting to OpenAI...")
    try:
        llm_client = get_client()
    except Exception:
        llm_client = None
        print("  → OpenAI unavailable, using fallback explanations.")

    # ── Load already-seen IDs ──────────────────────────────────────────────────
    seen_ids = load_seen_ids()

    # On first run, mark ALL existing emails as seen (don't scan old emails)
    if not seen_ids:
        print("\n[Init] First run — marking existing emails as already seen...")
        existing_ids = get_message_ids(gmail_service, max_results=50)
        seen_ids = set(existing_ids)
        save_seen_ids(seen_ids)
        print(f"[Init] {len(seen_ids)} existing emails marked as seen.")
        print("[Init] Watching for NEW emails from now on...\n")
    else:
        print(f"\n[Init] Loaded {len(seen_ids)} previously seen email IDs.")
        print("[Init] Watching for NEW emails...\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Checking inbox...", end=" ", flush=True)

            # Get latest email IDs
            latest_ids = get_message_ids(gmail_service, max_results=20)
            latest_set = set(latest_ids)

            # Find truly new emails (not seen before)
            new_ids = latest_set - seen_ids

            if new_ids:
                print(f"{len(new_ids)} NEW email(s) found!")

                for msg_id in new_ids:
                    try:
                        process_new_email(
                            gmail_service=gmail_service,
                            notify_service=notify_service,
                            msg_id=msg_id,
                            agent_chain=agent_chain,
                            rag=rag,
                            llm_client=llm_client,
                        )
                    except Exception as e:
                        print(f"  [Error] Failed to process {msg_id}: {e}")

                    # Mark as seen regardless of success
                    seen_ids.add(msg_id)

                # Save updated seen IDs
                save_seen_ids(seen_ids)

            else:
                print("No new emails.")

            # Wait before next check
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n[Stopped] Watcher stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            print(f"[Retry] Waiting {CHECK_INTERVAL} seconds before retry...")
            time.sleep(CHECK_INTERVAL)


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    watch()
