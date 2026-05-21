"""
agent.py
========
Agentic decision engine using LangChain with corrected imports.
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME     = "gpt-3.5-turbo"
TEMPERATURE    = 0

ANALYST_PROMPT = ChatPromptTemplate.from_template("""
You are a senior cybersecurity analyst specializing in email phishing detection.
Analyze the evidence below and make a final verdict. Think step-by-step.

════════════════════════════════════════
EMAIL EVIDENCE REPORT
════════════════════════════════════════

Subject      : {subject}
Sender       : {sender}
Sender Domain: {sender_domain}
Date         : {date}

── Automated Detection Scores ──
Rule-Based Score : {rule_score}/100
ML Model Score   : {ml_score}/100
Combined Risk    : {risk_score}/100
Risk Level       : {risk_level}

── Rule Flags Triggered ──
{rule_flags}

── Phishing Keywords Found ──
{keywords}

── URLs in Email ──
Total URLs      : {url_count}
Suspicious URLs : {suspicious_url_count}
Suspicious list : {suspicious_urls}
Domain Mismatch : {domain_mismatch}

── Email Body Preview ──
{body_preview}

════════════════════════════════════════

Analyze step by step:

STEP 1 - SENDER ANALYSIS: Is the sender domain legitimate or spoofed?
STEP 2 - CONTENT ANALYSIS: Any urgency triggers, threats, credential requests?
STEP 3 - LINK ANALYSIS: Suspicious URLs, IP addresses, domain mismatches?
STEP 4 - SCORE ANALYSIS: Is the risk score of {risk_score}/100 justified?
STEP 5 - FINAL VERDICT: Your conclusion based on all evidence.

Respond in this EXACT format:

VERDICT: [SAFE or PHISHING]
CONFIDENCE: [LOW or MEDIUM or HIGH]
AGENT_SCORE: [0-100]
REASONING:
[Your step-by-step reasoning, 3-6 sentences]
EXPLANATION:
[One plain-English sentence for a non-technical user]
""")


def parse_agent_response(raw_response: str) -> dict:
    result = {
        "verdict": "UNKNOWN", "confidence": "LOW",
        "agent_score": 50, "reasoning": "", "explanation": "",
    }
    current_section = None
    section_lines = []

    for line in raw_response.strip().split("\n"):
        s = line.strip()
        if s.startswith("VERDICT:"):
            val = s.replace("VERDICT:", "").strip().upper()
            result["verdict"] = val if val in ("SAFE", "PHISHING") else "UNKNOWN"
        elif s.startswith("CONFIDENCE:"):
            val = s.replace("CONFIDENCE:", "").strip().upper()
            result["confidence"] = val if val in ("LOW", "MEDIUM", "HIGH") else "LOW"
        elif s.startswith("AGENT_SCORE:"):
            try:
                result["agent_score"] = max(0, min(100, int(s.replace("AGENT_SCORE:", "").strip())))
            except ValueError:
                result["agent_score"] = 50
        elif s == "REASONING:":
            if current_section == "explanation":
                result["explanation"] = " ".join(section_lines).strip()
            current_section = "reasoning"
            section_lines = []
        elif s == "EXPLANATION:":
            if current_section == "reasoning":
                result["reasoning"] = " ".join(section_lines).strip()
            current_section = "explanation"
            section_lines = []
        elif current_section in ("reasoning", "explanation") and s:
            section_lines.append(s)

    if current_section == "explanation":
        result["explanation"] = " ".join(section_lines).strip()
    elif current_section == "reasoning":
        result["reasoning"] = " ".join(section_lines).strip()

    return result


def build_agent_chain():
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found.\n"
            "Make sure your .env file contains:\n"
            "  OPENAI_API_KEY=sk-proj-...your-key...\n"
        )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=OPENAI_API_KEY)
    return ANALYST_PROMPT | llm | StrOutputParser()


def format_list(items: list) -> str:
    return "None" if not items else "\n".join(f"  • {item}" for item in items)


def build_prompt_inputs(email: dict) -> dict:
    return {
        "subject":              email.get("subject", "(No Subject)"),
        "sender":               email.get("sender", "(Unknown)"),
        "sender_domain":        email.get("sender_domain", "(Unknown)"),
        "date":                 email.get("date", "(Unknown)"),
        "rule_score":           email.get("rule_score", 0),
        "ml_score":             email.get("ml_score", 0),
        "risk_score":           email.get("risk_score", 0),
        "risk_level":           email.get("risk_level", "LOW"),
        "rule_flags":           format_list(email.get("rule_flags", [])),
        "keywords":             format_list(email.get("keywords", [])),
        "url_count":            email.get("url_count", 0),
        "suspicious_url_count": email.get("suspicious_url_count", 0),
        "suspicious_urls":      format_list(email.get("suspicious_urls", [])),
        "domain_mismatch":      str(email.get("domain_mismatch", False)),
        "body_preview":         email.get("clean_body", "")[:500],
    }


def analyze_email(email: dict, chain) -> dict:
    try:
        raw = chain.invoke(build_prompt_inputs(email))
        agent_result = parse_agent_response(raw)
    except Exception as e:
        print(f"  [Agent Error] {e}")
        risk_score = email.get("risk_score", 0)
        agent_result = {
            "verdict":     "PHISHING" if risk_score >= 50 else "SAFE",
            "confidence":  "LOW",
            "agent_score": risk_score,
            "reasoning":   f"Agent unavailable. Using rule-based score: {risk_score}/100.",
            "explanation": f"This email received a risk score of {risk_score}/100 from automated rules.",
        }
    return {**email, **agent_result}


def analyze_all(detected_emails: list) -> list:
    print("\n[Agent] Building LangChain agent chain...")
    chain = build_agent_chain()
    print(f"[Agent] Analyzing {len(detected_emails)} emails with {MODEL_NAME}...\n")

    results = []
    for i, email in enumerate(detected_emails):
        subj = email.get("subject", "")[:55]
        print(f"[Agent] {i + 1}/{len(detected_emails)}: {subj}")
        result = analyze_email(email, chain)
        print(f"         → {result.get('verdict','?')} (confidence: {result.get('confidence','?')}, score: {result.get('agent_score',0)})")
        results.append(result)

    print(f"\n[Done] Agent analysis complete for {len(results)} emails.")
    return results


def save_agent_results(data: list, path: str = "agent_results.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Save] Agent results saved to '{path}'.")


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Email Safety Scanner — Phase 4: Agent")
    print("=" * 55)

    try:
        with open("detected.json", "r", encoding="utf-8") as f:
            detected_emails = json.load(f)
    except FileNotFoundError:
        print("[Error] detected.json not found. Run detector.py first.")
        exit(1)

    try:
        results = analyze_all(detected_emails)
    except ValueError as e:
        print(f"\n[Setup Error] {e}")
        exit(1)

    print("\n── Agent Verdict Summary ─────────────────────────────────")
    print(f"  {'Subject':<45} {'Verdict':<10} {'Score':>6}")
    print("  " + "-" * 65)
    for r in results:
        subj    = r.get("subject", "(No Subject)")[:44]
        verdict = r.get("verdict", "?")
        score   = r.get("agent_score", 0)
        print(f"  {subj:<45} {verdict:<10} {score:>5}%")

    save_agent_results(results)
