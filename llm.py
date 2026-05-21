"""
llm.py
======
LLM Reasoning Layer — generates plain-English explanations for each email.

What this file does:
  1. Takes the full analysis (rules + ML + agent + RAG) for each email
  2. Calls OpenAI GPT to generate a clear, human-readable explanation
  3. Explains WHY an email is safe or phishing in simple language
  4. Falls back to a rule-based explanation if OpenAI is unavailable

Example outputs:
  PHISHING: "This email is likely phishing because it uses urgent language
             to pressure you into clicking a suspicious link, and the sender
             domain does not match the links in the email body."

  SAFE:     "This email appears safe. It was sent from a verified LinkedIn
             domain, contains no suspicious links, and matches the typical
             pattern of a legitimate job alert notification."

Dependencies:
  openai, python-dotenv (already in requirements.txt)
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = "gpt-3.5-turbo"
TEMPERATURE    = 0.3   # Slight creativity for natural language, still focused


# ── OpenAI Client ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """Returns an OpenAI client. Raises ValueError if key not set."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file.\n"
            "Add: OPENAI_API_KEY=sk-proj-...your-key..."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ── Fallback Explanation (no OpenAI needed) ────────────────────────────────────

def generate_fallback_explanation(email: dict) -> str:
    """
    Generates a rule-based explanation without calling any LLM.
    Used when OpenAI is unavailable or quota is exceeded.

    Builds a natural sentence by combining the strongest signals found.

    Args:
        email: Fully enriched email dict (preprocessed + detected + agent + RAG)

    Returns:
        A plain-English explanation string.
    """
    verdict      = email.get("verdict", "UNKNOWN")
    risk_score   = email.get("risk_score", 0)
    rule_flags   = email.get("rule_flags", [])
    keywords     = email.get("keywords", [])
    sender_domain = email.get("sender_domain", "")
    suspicious_urls = email.get("suspicious_urls", [])
    domain_mismatch = email.get("domain_mismatch", False)

    # ── PHISHING explanation ───────────────────────────────────────────────────
    if verdict == "PHISHING":
        reasons = []

        if suspicious_urls:
            reasons.append("it contains suspicious links")
        if domain_mismatch:
            reasons.append("the links don't match the sender's domain")
        if any(k in keywords for k in ["urgent", "immediately", "act now", "expires"]):
            reasons.append("it uses urgent language to pressure you")
        if any(k in keywords for k in ["password", "verify", "login", "confirm your"]):
            reasons.append("it requests your login credentials")
        if any(k in keywords for k in ["prize", "winner", "lottery", "claim your"]):
            reasons.append("it makes unrealistic prize or reward claims")
        if any(k in keywords for k in ["will be terminated", "legal action", "suspended"]):
            reasons.append("it uses threats to create fear")
        if not sender_domain:
            reasons.append("the sender's identity could not be verified")

        # RAG context
        rag_matches = email.get("rag_top_matches", [])
        if rag_matches and rag_matches[0]["similarity"] > 0.3:
            top = rag_matches[0]
            reasons.append(
                f"it closely resembles a known {top['attack_type']} "
                f"({top['similarity']*100:.0f}% similarity)"
            )

        if reasons:
            reason_str = ", and ".join(reasons[:3])  # Keep it concise
            return (
                f"This email is likely phishing because {reason_str}. "
                f"Risk score: {risk_score}/100. Do not click any links or provide personal information."
            )
        else:
            return (
                f"This email has been flagged as potential phishing with a risk score of "
                f"{risk_score}/100. Exercise caution before clicking any links."
            )

    # ── SAFE explanation ───────────────────────────────────────────────────────
    elif verdict == "SAFE":
        safe_signals = []

        if sender_domain:
            safe_signals.append(f"sent from a recognizable domain ({sender_domain})")
        if not suspicious_urls:
            safe_signals.append("contains no suspicious links")
        if not keywords:
            safe_signals.append("uses no phishing keywords")
        if not domain_mismatch:
            safe_signals.append("links match the sender's domain")

        if safe_signals:
            signal_str = ", ".join(safe_signals[:3])
            return (
                f"This email appears safe. It is {signal_str}. "
                f"Risk score: {risk_score}/100."
            )
        else:
            return (
                f"This email appears safe with a low risk score of {risk_score}/100. "
                f"No significant phishing indicators were detected."
            )

    # ── Unknown ────────────────────────────────────────────────────────────────
    else:
        return (
            f"This email could not be classified with confidence. "
            f"Risk score: {risk_score}/100. Review it carefully before taking any action."
        )


# ── LLM Prompt Builder ─────────────────────────────────────────────────────────

def build_explanation_prompt(email: dict) -> str:
    """
    Builds a concise prompt asking the LLM to explain the analysis result.

    Args:
        email: Fully enriched email dict.

    Returns:
        Prompt string for the LLM.
    """
    verdict       = email.get("verdict", "UNKNOWN")
    risk_score    = email.get("risk_score", 0)
    rule_flags    = email.get("rule_flags", [])
    keywords      = email.get("keywords", [])
    rag_context   = email.get("rag_context", "No similar examples found.")
    subject       = email.get("subject", "")
    sender        = email.get("sender", "")
    sender_domain = email.get("sender_domain", "")
    reasoning     = email.get("reasoning", "")

    flags_str    = "\n".join(f"  • {f}" for f in rule_flags) if rule_flags else "  None"
    keywords_str = ", ".join(keywords[:8]) if keywords else "None"

    return f"""
You are a cybersecurity assistant explaining email analysis results to a non-technical user.

EMAIL SUMMARY:
  Subject      : {subject}
  Sender       : {sender}
  Sender Domain: {sender_domain}
  Verdict      : {verdict}
  Risk Score   : {risk_score}/100

DETECTION EVIDENCE:
  Rule Flags:
{flags_str}
  Keywords Found : {keywords_str}
  RAG Context    : {rag_context}
  Agent Reasoning: {reasoning[:300] if reasoning else 'Not available'}

YOUR TASK:
Write a clear, concise explanation (2-3 sentences) for a non-technical user explaining:
  1. Whether this email is safe or phishing
  2. The main reason(s) why
  3. What the user should do

Rules:
  - Use simple language (no technical jargon)
  - Be direct and specific — mention actual signals found
  - If PHISHING: warn the user clearly
  - If SAFE: reassure but remind them to stay cautious
  - Do NOT start with "I" — start with "This email..."

Write only the explanation. No headings, no bullet points, just 2-3 sentences.
""".strip()


# ── Main LLM Explanation Function ─────────────────────────────────────────────

def generate_explanation(email: dict, client: OpenAI = None) -> str:
    """
    Generates a plain-English explanation for a single email using GPT.

    Falls back to rule-based explanation if OpenAI call fails.

    Args:
        email:  Fully enriched email dict.
        client: OpenAI client (created if not provided).

    Returns:
        Explanation string.
    """
    if client is None:
        try:
            client = get_client()
        except ValueError:
            return generate_fallback_explanation(email)

    prompt = build_explanation_prompt(email)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful cybersecurity assistant. "
                        "Explain email safety analysis results in plain English. "
                        "Be concise, clear, and helpful."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=150,   # Keep explanations short and punchy
        )
        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg:
            print(f"  [LLM] OpenAI quota exceeded — using fallback explanation.")
        else:
            print(f"  [LLM] OpenAI error: {e} — using fallback explanation.")
        return generate_fallback_explanation(email)


def generate_all_explanations(emails: list[dict]) -> list[dict]:
    """
    Generates explanations for all emails.

    Creates the OpenAI client once and reuses it for all emails
    (more efficient than creating a new client per email).

    Args:
        emails: List of enriched email dicts.

    Returns:
        List of email dicts with 'llm_explanation' field added.
    """
    print(f"\n[LLM] Generating explanations for {len(emails)} emails...")

    # Try to create OpenAI client once
    try:
        client = get_client()
        print(f"[LLM] Using OpenAI {MODEL_NAME} for explanations.")
    except ValueError:
        client = None
        print("[LLM] OpenAI not available — using rule-based fallback explanations.")

    results = []
    for i, email in enumerate(emails):
        subj    = email.get("subject", "")[:50]
        verdict = email.get("verdict", "?")
        print(f"[LLM] {i+1}/{len(emails)}: {subj[:45]} [{verdict}]")

        explanation = generate_explanation(email, client)

        # Add explanation to email dict
        enriched = {**email, "llm_explanation": explanation}
        results.append(enriched)

        # Print the explanation
        print(f"         → {explanation[:100]}...")

    print(f"\n[Done] Explanations generated for {len(results)} emails.")
    return results


# ── Save Helper ────────────────────────────────────────────────────────────────

def save_explained_results(data: list[dict], path: str = "final_results.json") -> None:
    """
    Saves the fully explained results to JSON.
    This is the final output used by the dashboard (Phase 8).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Save] Final results saved to '{path}'.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AI Email Safety Scanner — Phase 6: LLM Reasoning")
    print("=" * 55)

    # Load RAG-enriched results from Phase 5
    try:
        with open("rag_results.json", "r", encoding="utf-8") as f:
            emails = json.load(f)
    except FileNotFoundError:
        print("[Error] rag_results.json not found. Run rag.py first.")
        exit(1)

    # Generate explanations
    results = generate_all_explanations(emails)

    # Print full explanation for each email
    print("\n── Final Explanations ────────────────────────────────────")
    for r in results:
        subj        = r.get("subject", "")[:55]
        verdict     = r.get("verdict", "?")
        score       = r.get("risk_score", 0)
        explanation = r.get("llm_explanation", "")

        print(f"\n  [{verdict}] {subj}")
        print(f"  Score  : {score}/100")
        print(f"  Explain: {explanation}")

    # Save final results for Phase 7 and Phase 8
    save_explained_results(results)
