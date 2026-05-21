"""
app.py
======
Email Inbox View — displays your emails like a real inbox
with SAFE / PHISHING labels shown directly on each email.

Run with:
  streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Email Inbox",
    page_icon="📧",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.inbox-row {
    background: #1a1d27;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-left: 5px solid #444;
}
.inbox-phishing { border-left-color: #ff4b4b; }
.inbox-safe     { border-left-color: #00c853; }

.label-phishing {
    background: #ff4b4b22;
    color: #ff4b4b;
    border: 1px solid #ff4b4b66;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    margin-left: 8px;
}
.label-safe {
    background: #00c85322;
    color: #00c853;
    border: 1px solid #00c85366;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    margin-left: 8px;
}
.score-pill {
    background: #2a2d3a;
    color: #aaa;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    margin-left: 6px;
}
.subject-text {
    font-size: 15px;
    font-weight: 600;
    color: #eee;
    display: inline;
}
.sender-text {
    font-size: 12px;
    color: #666;
    margin-top: 3px;
}
.explain-text {
    font-size: 13px;
    color: #999;
    margin-top: 6px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Load results ───────────────────────────────────────────────────────────────
def load_results():
    path = Path("output.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## 📧 Your Inbox")
st.markdown("Each email is labeled **SAFE** or **PHISHING** by the AI scanner.")
st.markdown("---")

results = load_results()

if not results:
    st.warning("No results found. Run the pipeline first:")
    st.code("python gmail_fetcher.py\npython preprocess.py\npython detector.py\npython agent.py\npython rag.py\npython llm.py\npython output.py")
    st.stop()

# ── Filter bar ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    search = st.text_input("🔍", placeholder="Search subject or sender...", label_visibility="collapsed")
with col2:
    filt = st.selectbox("", ["All", "Phishing Only", "Safe Only"], label_visibility="collapsed")

filtered = results
if search:
    filtered = [r for r in filtered if
                search.lower() in r.get("subject","").lower() or
                search.lower() in r.get("sender","").lower()]
if filt == "Phishing Only":
    filtered = [r for r in filtered if r["classification"] == "PHISHING"]
elif filt == "Safe Only":
    filtered = [r for r in filtered if r["classification"] == "SAFE"]

# ── Stats row ─────────────────────────────────────────────────────────────────
total    = len(results)
phishing = sum(1 for r in results if r["classification"] == "PHISHING")
safe     = total - phishing

c1, c2, c3 = st.columns(3)
c1.metric("📬 Total", total)
c2.metric("⚠️ Phishing", phishing)
c3.metric("✅ Safe", safe)
st.markdown("---")

# ── Email list ─────────────────────────────────────────────────────────────────
st.markdown(f"**{len(filtered)} emails**")

for email in filtered:
    classification = email.get("classification", "UNKNOWN")
    subject        = email.get("subject", "(No Subject)")
    sender         = email.get("sender", "(Unknown)")
    date           = email.get("date", "")[:25]
    risk_score     = email.get("risk_score", 0)
    explanation    = email.get("explanation", "")
    reasons        = email.get("reasons", [])

    is_phishing = classification == "PHISHING"
    row_class   = "inbox-phishing" if is_phishing else "inbox-safe"
    label_class = "label-phishing" if is_phishing else "label-safe"
    label_text  = "⚠ PHISHING" if is_phishing else "✓ SAFE"

    # Render email row
    st.markdown(f"""
    <div class="inbox-row {row_class}">
        <div>
            <span class="subject-text">{subject[:65]}</span>
            <span class="{label_class}">{label_text}</span>
            <span class="score-pill">Risk: {risk_score}/100</span>
        </div>
        <div class="sender-text">From: {sender[:60]} &nbsp;·&nbsp; {date}</div>
        <div class="explain-text">{explanation[:120]}...</div>
    </div>
    """, unsafe_allow_html=True)

    # Expandable details
    with st.expander("View full analysis"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Risk Score", f"{risk_score}/100")
        col_b.metric("Risk Level", email.get("risk_level", "LOW"))
        col_c.metric("Confidence", email.get("confidence", "LOW"))

        st.markdown("**🤖 AI Explanation**")
        st.info(explanation)

        if reasons:
            st.markdown("**🚩 Evidence Found**")
            for r in reasons:
                color = "🔴" if is_phishing else "🟢"
                st.markdown(f"{color} {r}")

        meta = email.get("metadata", {})
        sus_urls = meta.get("suspicious_urls", [])
        if sus_urls:
            st.markdown("**🔗 Suspicious URLs**")
            for url in sus_urls[:3]:
                st.code(url)

        keywords = meta.get("keywords_found", [])
        if keywords:
            st.markdown("**🔑 Phishing Keywords**")
            st.markdown(" · ".join(f"`{k}`" for k in keywords[:8]))

        rag = meta.get("rag_top_match")
        if rag and rag.get("similarity", 0) > 0.15:
            st.markdown("**📚 Similar Known Attack**")
            st.warning(
                f"**{rag['subject']}**\n\n"
                f"Type: {rag['attack_type']} · "
                f"Similarity: {rag['similarity']*100:.0f}% · "
                f"Known Risk: {rag['risk_score']}/100"
            )
