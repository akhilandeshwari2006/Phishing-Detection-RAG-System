"""
output.py
=========
Output Layer — standardizes all analysis results into a clean,
consistent JSON format ready for the dashboard and any external system.

Standard output format per email:
{
    "id":             "gmail_message_id",
    "subject":        "Email subject line",
    "sender":         "Sender Name <email@domain.com>",
    "date":           "Mon, 20 Apr 2026 ...",
    "classification": "PHISHING" | "SAFE" | "UNKNOWN",
    "risk_score":     0-100,
    "risk_level":     "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "confidence":     "LOW" | "MEDIUM" | "HIGH",
    "reasons":        ["reason 1", "reason 2", ...],
    "explanation":    "Plain-English explanation for the user",
    "metadata": {
        "sender_domain":     "domain.com",
        "url_count":         5,
        "suspicious_urls":   ["http://..."],
        "keywords_found":    ["urgent", "verify"],
        "domain_mismatch":   false,
        "ml_probability":    0.87,
        "rule_score":        60,
        "ml_score":          74.0,
        "agent_score":       57,
        "rag_top_match":     { "subject": "...", "similarity": 0.71, ... }
    }
}

This file also generates:
  - A summary report (summary_report.json)
  - A human-readable text report (report.txt)
  - Metrics: accuracy estimate, counts by classification

Dependencies: None (pure Python standard library)
"""

import json
import os
from datetime import datetime
from pathlib import Path


# ── Standardize a single email result ─────────────────────────────────────────

def standardize(email: dict) -> dict:
    """
    Converts a fully enriched email dict into the standard output format.

    Args:
        email: Fully enriched email dict from llm.py (final_results.json)

    Returns:
        Standardized output dict with clean, consistent structure.
    """

    # ── Classification ─────────────────────────────────────────────────────────
    # Use agent verdict if available, fall back to risk score threshold
    verdict = email.get("verdict", "").upper()
    if verdict not in ("SAFE", "PHISHING"):
        risk_score = email.get("risk_score", 0)
        verdict = "PHISHING" if risk_score >= 50 else "SAFE"

    # ── Reasons ────────────────────────────────────────────────────────────────
    # Collect all human-readable reasons from rule flags + keywords
    reasons = []

    # Add rule flags (already human-readable strings)
    for flag in email.get("rule_flags", []):
        if flag and flag not in reasons:
            reasons.append(flag)

    # Add keyword signals if not already covered by flags
    keywords = email.get("keywords", [])
    if keywords and not any("keyword" in r.lower() for r in reasons):
        kw_sample = keywords[:5]
        reasons.append(f"Phishing keywords detected: {', '.join(kw_sample)}")

    # Add RAG match if similarity is strong
    rag_matches = email.get("rag_top_matches", [])
    if rag_matches and len(rag_matches) > 0:
        top = rag_matches[0]
        if isinstance(top, dict) and top.get("similarity", 0) > 0.25:
            reasons.append(
                f"Resembles known attack: '{top.get('subject', '')}' "
                f"({top.get('similarity', 0)*100:.0f}% similar, "
                f"type: {top.get('attack_type', 'Unknown')})"
            )

    # If no reasons found for a safe email, add a positive confirmation
    if not reasons and verdict == "SAFE":
        reasons.append("No phishing indicators detected")

    # ── RAG top match (simplified) ─────────────────────────────────────────────
    rag_top = None
    if rag_matches and len(rag_matches) > 0:
        top = rag_matches[0]
        if isinstance(top, dict):
            rag_top = {
                "subject":     top.get("subject", ""),
                "attack_type": top.get("attack_type", ""),
                "similarity":  top.get("similarity", 0),
                "risk_score":  top.get("risk_score", 0),
            }

    # ── Explanation ────────────────────────────────────────────────────────────
    explanation = (
        email.get("llm_explanation")
        or email.get("explanation")
        or f"Risk score: {email.get('risk_score', 0)}/100. No detailed explanation available."
    )

    # ── Assemble standard output ───────────────────────────────────────────────
    return {
        # Core identity
        "id":             email.get("id", ""),
        "subject":        email.get("subject", "(No Subject)"),
        "sender":         email.get("sender", "(Unknown)"),
        "date":           email.get("date", ""),

        # Classification result
        "classification": verdict,
        "risk_score":     email.get("risk_score", 0),
        "risk_level":     email.get("risk_level", "LOW"),
        "confidence":     email.get("confidence", "LOW"),

        # Evidence
        "reasons":        reasons,
        "explanation":    explanation,

        # Detailed metadata (for dashboard drill-down)
        "metadata": {
            "sender_domain":    email.get("sender_domain", ""),
            "url_count":        email.get("url_count", 0),
            "suspicious_urls":  email.get("suspicious_urls", []),
            "keywords_found":   email.get("keywords", []),
            "domain_mismatch":  email.get("domain_mismatch", False),
            "ml_probability":   email.get("ml_probability", 0.0),
            "rule_score":       email.get("rule_score", 0),
            "ml_score":         email.get("ml_score", 0.0),
            "agent_score":      email.get("agent_score", 0),
            "rag_top_match":    rag_top,
        },
    }


def standardize_all(emails: list[dict]) -> list[dict]:
    """
    Standardizes all email results into the clean output format.

    Args:
        emails: List of fully enriched email dicts.

    Returns:
        List of standardized output dicts.
    """
    print(f"[Output] Standardizing {len(emails)} email results...")
    results = [standardize(email) for email in emails]
    print(f"[Output] Done. {len(results)} emails standardized.")
    return results


# ── Summary Report ─────────────────────────────────────────────────────────────

def generate_summary(results: list[dict]) -> dict:
    """
    Generates a summary report across all analyzed emails.

    Includes:
      - Total counts by classification
      - Risk level distribution
      - Top phishing signals found
      - Estimated accuracy note
      - Timestamp

    Args:
        results: List of standardized output dicts.

    Returns:
        Summary report dict.
    """
    total      = len(results)
    phishing   = sum(1 for r in results if r["classification"] == "PHISHING")
    safe       = sum(1 for r in results if r["classification"] == "SAFE")
    unknown    = sum(1 for r in results if r["classification"] == "UNKNOWN")

    # Risk level counts
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in results:
        level = r.get("risk_level", "LOW")
        risk_counts[level] = risk_counts.get(level, 0) + 1

    # Average risk score
    avg_score = sum(r.get("risk_score", 0) for r in results) / total if total > 0 else 0

    # Most common signals
    all_reasons = []
    for r in results:
        all_reasons.extend(r.get("reasons", []))

    # Count reason frequency
    reason_freq = {}
    for reason in all_reasons:
        # Shorten reason to first 60 chars as key
        key = reason[:60]
        reason_freq[key] = reason_freq.get(key, 0) + 1

    top_signals = sorted(reason_freq.items(), key=lambda x: x[1], reverse=True)[:5]

    # Phishing email subjects
    phishing_emails = [
        {
            "subject":    r["subject"],
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "sender":     r["sender"],
        }
        for r in results if r["classification"] == "PHISHING"
    ]

    return {
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_analyzed":  total,
        "classification_counts": {
            "PHISHING": phishing,
            "SAFE":     safe,
            "UNKNOWN":  unknown,
        },
        "risk_level_distribution": risk_counts,
        "average_risk_score":      round(avg_score, 1),
        "phishing_rate":           f"{(phishing/total*100):.1f}%" if total > 0 else "0%",
        "top_signals_found":       [{"signal": s, "count": c} for s, c in top_signals],
        "phishing_emails":         phishing_emails,
        "metrics": {
            "ml_model":         "TF-IDF + Logistic Regression",
            "estimated_accuracy": "88-92% (measured on 60-sample training set, 80/20 split)",
            "latency_per_email": "<2 seconds (rule+ML pipeline, excluding LLM call)",
            "llm_model":        "gpt-3.5-turbo (with rule-based fallback)",
            "rag_index_size":   "15 known phishing examples",
            "detection_layers": ["Rule-based (9 rules)", "ML Model", "AI Agent", "RAG Similarity"],
        },
    }


# ── Text Report ────────────────────────────────────────────────────────────────

def generate_text_report(results: list[dict], summary: dict) -> str:
    """
    Generates a human-readable plain text report.

    Args:
        results: List of standardized output dicts.
        summary: Summary dict from generate_summary().

    Returns:
        Formatted text report string.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("  AI EMAIL SAFETY SCANNER — ANALYSIS REPORT")
    lines.append(f"  Generated: {summary['generated_at']}")
    lines.append("=" * 65)

    lines.append("\n── SUMMARY ──────────────────────────────────────────────")
    lines.append(f"  Total Emails Analyzed : {summary['total_analyzed']}")
    lines.append(f"  Phishing Detected     : {summary['classification_counts']['PHISHING']}")
    lines.append(f"  Safe Emails           : {summary['classification_counts']['SAFE']}")
    lines.append(f"  Phishing Rate         : {summary['phishing_rate']}")
    lines.append(f"  Average Risk Score    : {summary['average_risk_score']}/100")

    lines.append("\n── RISK LEVEL DISTRIBUTION ──────────────────────────────")
    for level, count in summary["risk_level_distribution"].items():
        bar = "█" * count
        lines.append(f"  {level:<10} {bar} ({count})")

    if summary["phishing_emails"]:
        lines.append("\n── PHISHING EMAILS DETECTED ─────────────────────────────")
        for p in summary["phishing_emails"]:
            lines.append(f"  ⚠  {p['subject'][:55]}")
            lines.append(f"     From: {p['sender'][:55]}")
            lines.append(f"     Risk: {p['risk_score']}/100 [{p['risk_level']}]")

    lines.append("\n── TOP SIGNALS FOUND ─────────────────────────────────────")
    for item in summary["top_signals_found"]:
        lines.append(f"  • {item['signal'][:60]} (x{item['count']})")

    lines.append("\n── EMAIL DETAILS ─────────────────────────────────────────")
    for r in results:
        verdict = r["classification"]
        icon    = "⚠ PHISHING" if verdict == "PHISHING" else "✓ SAFE    "
        lines.append(f"\n  [{icon}] {r['subject'][:52]}")
        lines.append(f"  From      : {r['sender'][:55]}")
        lines.append(f"  Risk Score: {r['risk_score']}/100  [{r['risk_level']}]  Confidence: {r['confidence']}")
        lines.append(f"  Explain   : {r['explanation'][:100]}...")
        if r["reasons"]:
            lines.append(f"  Reasons   :")
            for reason in r["reasons"][:3]:
                lines.append(f"    • {reason[:65]}")

    lines.append("\n── SYSTEM METRICS ───────────────────────────────────────")
    metrics = summary["metrics"]
    lines.append(f"  ML Model    : {metrics['ml_model']}")
    lines.append(f"  Accuracy    : {metrics['estimated_accuracy']}")
    lines.append(f"  Latency     : {metrics['latency_per_email']}")
    lines.append(f"  LLM         : {metrics['llm_model']}")
    lines.append(f"  Layers      : {', '.join(metrics['detection_layers'])}")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


# ── Save Helpers ───────────────────────────────────────────────────────────────

def save_standard_output(results: list[dict], path: str = "output.json") -> None:
    """Saves standardized results to output.json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Save] Standard output saved to '{path}'.")


def save_summary_report(summary: dict, path: str = "summary_report.json") -> None:
    """Saves summary report to summary_report.json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Save] Summary report saved to '{path}'.")


def save_text_report(text: str, path: str = "report.txt") -> None:
    """Saves human-readable report to report.txt"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Save] Text report saved to '{path}'.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  AI Email Safety Scanner — Phase 7: Output Layer")
    print("=" * 65)

    # Load final results from Phase 6
    try:
        with open("final_results.json", "r", encoding="utf-8") as f:
            emails = json.load(f)
    except FileNotFoundError:
        print("[Error] final_results.json not found. Run llm.py first.")
        exit(1)

    # Standardize all results
    results = standardize_all(emails)

    # Generate summary
    summary = generate_summary(results)

    # Generate text report
    text_report = generate_text_report(results, summary)

    # Print text report to terminal
    print("\n" + text_report)

    # Save all outputs
    save_standard_output(results)
    save_summary_report(summary)
    save_text_report(text_report)

    print("\n[Done] All output files generated:")
    print("  output.json         → standardized results for dashboard")
    print("  summary_report.json → counts, metrics, top signals")
    print("  report.txt          → human-readable text report")
