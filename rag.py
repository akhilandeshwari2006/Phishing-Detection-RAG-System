"""
rag.py
======
RAG (Retrieval Augmented Generation) system using FAISS vector store.

What this file does:
  1. Stores known phishing email examples as vector embeddings
  2. When a new email comes in, finds the most similar past examples
  3. Returns those similar examples to help the agent make better decisions
  4. Uses TF-IDF vectors (no OpenAI embeddings needed — works offline!)

How RAG improves detection:
  - "This email looks 87% similar to a known PayPal phishing email"
  - Gives the agent real evidence from past attacks
  - Catches new phishing emails that look like old ones

Flow:
  new email → vectorize → FAISS search → top-K similar examples
  → pass examples to agent as extra context

Dependencies:
  faiss-cpu, scikit-learn, numpy, joblib (already in requirements.txt)
"""

import json
import numpy as np
import joblib
import faiss
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Known Phishing Examples Database ──────────────────────────────────────────
# These are stored in the FAISS index.
# In production, you'd grow this database over time as new phishing
# emails are confirmed and added by analysts.

PHISHING_EXAMPLES = [
    {
        "id": "phish_001",
        "subject": "Urgent: Your account has been suspended",
        "text": "Dear customer, your account has been suspended due to unusual activity. "
                "Please verify your identity immediately by clicking the link below. "
                "Failure to do so will result in permanent account closure.",
        "attack_type": "Account Suspension Scam",
        "target": "Generic / Banking",
        "risk_score": 92,
    },
    {
        "id": "phish_002",
        "subject": "Your PayPal account is limited",
        "text": "We have limited your PayPal account. To restore full access, "
                "please confirm your identity and update your billing information. "
                "Click here to verify your account now.",
        "attack_type": "PayPal Phishing",
        "target": "PayPal Users",
        "risk_score": 95,
    },
    {
        "id": "phish_003",
        "subject": "Action Required: Update your payment information",
        "text": "Your Netflix membership has been suspended due to a problem with "
                "your last payment. Update your payment information to continue "
                "your membership. Click the button below to update your details.",
        "attack_type": "Netflix Billing Scam",
        "target": "Netflix Subscribers",
        "risk_score": 88,
    },
    {
        "id": "phish_004",
        "subject": "Congratulations! You have won a prize",
        "text": "You have been selected as our lucky winner! Claim your $1000 prize now. "
                "This offer expires in 24 hours. Click here to claim your reward. "
                "Provide your bank details to receive the transfer.",
        "attack_type": "Prize / Lottery Scam",
        "target": "General Public",
        "risk_score": 97,
    },
    {
        "id": "phish_005",
        "subject": "IRS Tax Refund Notification",
        "text": "After the last annual calculation of your fiscal activity, we have "
                "determined that you are eligible to receive a tax refund of $347.50. "
                "Please submit the tax refund request by clicking here.",
        "attack_type": "IRS / Tax Refund Scam",
        "target": "US Taxpayers",
        "risk_score": 94,
    },
    {
        "id": "phish_006",
        "subject": "Security Alert: Suspicious login attempt",
        "text": "We detected a suspicious login attempt on your account from an "
                "unknown device. If this was not you, please secure your account "
                "immediately by verifying your password and personal details.",
        "attack_type": "Security Alert Phishing",
        "target": "Generic Account Holders",
        "risk_score": 85,
    },
    {
        "id": "phish_007",
        "subject": "Your Microsoft account will be deleted",
        "text": "Your Microsoft account is scheduled for deletion due to inactivity. "
                "To prevent this, please verify your account within 48 hours. "
                "Enter your username and password to confirm your identity.",
        "attack_type": "Microsoft Account Phishing",
        "target": "Microsoft Users",
        "risk_score": 90,
    },
    {
        "id": "phish_008",
        "subject": "Package delivery failed - action required",
        "text": "We attempted to deliver your package but were unable to complete "
                "the delivery. Please click the link below to reschedule your "
                "delivery and confirm your address and payment details.",
        "attack_type": "Delivery / Shipping Scam",
        "target": "Online Shoppers",
        "risk_score": 82,
    },
    {
        "id": "phish_009",
        "subject": "Your Apple ID has been locked",
        "text": "Your Apple ID has been locked for security reasons. "
                "Someone tried to sign in to your account from an unrecognized device. "
                "Please verify your identity to unlock your account immediately.",
        "attack_type": "Apple ID Phishing",
        "target": "Apple Users",
        "risk_score": 91,
    },
    {
        "id": "phish_010",
        "subject": "Invoice overdue - immediate payment required",
        "text": "This is a final notice regarding your overdue invoice. "
                "Legal action will be taken if payment is not received within 24 hours. "
                "Click here to pay now and avoid legal proceedings.",
        "attack_type": "Fake Invoice / Legal Threat",
        "target": "Business Users",
        "risk_score": 89,
    },
    {
        "id": "phish_011",
        "subject": "Your bank account requires verification",
        "text": "Dear account holder, we have detected unusual transactions on your "
                "bank account. For your security, we have temporarily suspended your "
                "online access. Please login to verify your identity and restore access.",
        "attack_type": "Bank Phishing",
        "target": "Banking Customers",
        "risk_score": 93,
    },
    {
        "id": "phish_012",
        "subject": "Crypto wallet: Urgent action needed",
        "text": "Your cryptocurrency wallet has been flagged for suspicious activity. "
                "To secure your funds, please verify your wallet credentials immediately. "
                "Enter your seed phrase to confirm ownership of your wallet.",
        "attack_type": "Crypto Wallet Scam",
        "target": "Crypto Users",
        "risk_score": 98,
    },
    {
        "id": "phish_013",
        "subject": "HR: Important update to your salary details",
        "text": "Dear employee, we need to update your salary payment details in our system. "
                "Please provide your bank account number and sort code to ensure "
                "your next salary payment is processed correctly.",
        "attack_type": "HR / Payroll Scam",
        "target": "Corporate Employees",
        "risk_score": 87,
    },
    {
        "id": "phish_014",
        "subject": "COVID-19 relief fund - claim your payment",
        "text": "You are eligible for a COVID-19 government relief payment of $1,200. "
                "To claim your payment, please provide your social security number "
                "and bank account details using the secure form below.",
        "attack_type": "Government Relief Scam",
        "target": "General Public",
        "risk_score": 96,
    },
    {
        "id": "phish_015",
        "subject": "Your password will expire today",
        "text": "Your email password will expire in 2 hours. To keep your current password "
                "and avoid losing access to your account, please click the link below "
                "and enter your current password to continue.",
        "attack_type": "Password Expiry Phishing",
        "target": "Email Users",
        "risk_score": 84,
    },
]


# ── File paths for saved index ─────────────────────────────────────────────────

VECTORIZER_PATH = "rag_vectorizer.pkl"
INDEX_PATH      = "rag_index.faiss"
EXAMPLES_PATH   = "rag_examples.json"


# ── RAG System Class ───────────────────────────────────────────────────────────

class PhishingRAG:
    """
    RAG system for phishing email similarity search using FAISS.

    Attributes:
        vectorizer: TF-IDF vectorizer fitted on phishing examples.
        index:      FAISS flat L2 index for fast similarity search.
        examples:   List of phishing example dicts.
    """

    def __init__(self):
        self.vectorizer = None
        self.index      = None
        self.examples   = []

    def build(self, examples: list[dict] = None) -> None:
        """
        Builds the FAISS index from phishing examples.

        Steps:
          1. Fit TF-IDF vectorizer on all example texts
          2. Convert texts to dense float32 vectors
          3. Create FAISS flat L2 index
          4. Add all vectors to the index
          5. Save everything to disk

        Args:
            examples: List of phishing example dicts (uses PHISHING_EXAMPLES if None).
        """
        if examples is None:
            examples = PHISHING_EXAMPLES

        self.examples = examples
        texts = [f"{ex['subject']} {ex['text']}" for ex in examples]

        print(f"[RAG] Building index from {len(texts)} phishing examples...")

        # Step 1: Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            stop_words="english",
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Step 2: Convert sparse matrix to dense float32
        # FAISS requires dense float32 arrays
        vectors = tfidf_matrix.toarray().astype(np.float32)

        # Step 3: Create FAISS index
        # IndexFlatL2 = exact nearest neighbour search using L2 (Euclidean) distance
        # For production with millions of examples, use IndexIVFFlat instead
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)

        # Step 4: Add vectors to index
        self.index.add(vectors)

        print(f"[RAG] Index built. {self.index.ntotal} vectors, dimension={dimension}.")

        # Step 5: Save to disk
        self._save()

    def _save(self) -> None:
        """Saves vectorizer, FAISS index, and examples to disk."""
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        faiss.write_index(self.index, INDEX_PATH)
        with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
            json.dump(self.examples, f, indent=2)
        print(f"[RAG] Saved: {VECTORIZER_PATH}, {INDEX_PATH}, {EXAMPLES_PATH}")

    def load(self) -> bool:
        """
        Loads a previously saved RAG index from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        if not all(Path(p).exists() for p in [VECTORIZER_PATH, INDEX_PATH, EXAMPLES_PATH]):
            return False

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.index      = faiss.read_index(INDEX_PATH)
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        print(f"[RAG] Loaded index: {self.index.ntotal} vectors.")
        return True

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Finds the top-K most similar phishing examples for a given email text.

        Args:
            query_text: The email text to search for (subject + body).
            top_k:      Number of similar examples to return (default 3).

        Returns:
            List of dicts, each containing:
              - example:    The matched phishing example dict
              - similarity: Similarity score 0.0–1.0 (1.0 = identical)
              - distance:   Raw L2 distance (lower = more similar)
        """
        if not query_text or not query_text.strip():
            return []

        # Vectorize the query using the fitted TF-IDF vectorizer
        query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)

        # Search FAISS index — returns distances and indices of nearest neighbours
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid results
                continue

            # Convert L2 distance to a similarity score (0–1)
            # L2 distance of 0 = identical, larger = more different
            # We use exponential decay: similarity = e^(-distance)
            similarity = float(np.exp(-dist))

            results.append({
                "example":    self.examples[idx],
                "similarity": round(similarity, 4),
                "distance":   round(float(dist), 4),
            })

        # Sort by similarity descending (most similar first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def enrich_email(self, email: dict, top_k: int = 3) -> dict:
        """
        Enriches an email dict with RAG similarity results.

        Adds a 'rag_matches' key containing the top similar phishing examples,
        and a 'rag_context' string that can be injected into the agent prompt.

        Args:
            email: Any email dict with 'full_text' or 'clean_body' field.
            top_k: Number of similar examples to retrieve.

        Returns:
            The email dict with added 'rag_matches' and 'rag_context' fields.
        """
        query_text = email.get("full_text") or email.get("clean_body", "")
        matches    = self.search(query_text, top_k=top_k)

        # Build a readable context string for the agent
        if matches:
            context_lines = ["Similar known phishing emails found:"]
            for i, match in enumerate(matches):
                ex   = match["example"]
                sim  = match["similarity"]
                context_lines.append(
                    f"  [{i+1}] {sim*100:.0f}% similar — '{ex['subject']}' "
                    f"(Type: {ex['attack_type']}, Risk: {ex['risk_score']}/100)"
                )
            rag_context = "\n".join(context_lines)
        else:
            rag_context = "No similar phishing examples found in database."

        return {
            **email,
            "rag_matches": matches,
            "rag_context": rag_context,
        }

    def enrich_all(self, emails: list[dict], top_k: int = 3) -> list[dict]:
        """
        Enriches all emails with RAG similarity results.

        Args:
            emails: List of email dicts.
            top_k:  Number of similar examples per email.

        Returns:
            List of enriched email dicts.
        """
        print(f"\n[RAG] Enriching {len(emails)} emails with similarity search...")
        enriched = []
        for i, email in enumerate(emails):
            subj = email.get("subject", "")[:55]
            result = self.enrich_email(email, top_k=top_k)
            top_match = result["rag_matches"][0] if result["rag_matches"] else None
            sim_str = f"{top_match['similarity']*100:.0f}% match" if top_match else "no match"
            print(f"[RAG] {i+1}/{len(emails)}: {subj[:45]} → {sim_str}")
            enriched.append(result)

        print(f"[Done] RAG enrichment complete.")
        return enriched


# ── Load or Build helper ───────────────────────────────────────────────────────

def get_rag() -> PhishingRAG:
    """
    Returns a ready-to-use RAG system.
    Loads from disk if available, otherwise builds fresh.
    """
    rag = PhishingRAG()
    if not rag.load():
        print("[RAG] No saved index found. Building fresh index...")
        rag.build()
    return rag


# ── Save helper ────────────────────────────────────────────────────────────────

def save_rag_results(data: list[dict], path: str = "rag_results.json") -> None:
    """Saves RAG-enriched results to JSON (excludes heavy vector data)."""
    # Remove numpy arrays before saving to JSON
    clean_data = []
    for email in data:
        clean_email = {k: v for k, v in email.items() if k != "rag_matches"}
        # Save simplified match info
        matches = email.get("rag_matches", [])
        clean_email["rag_top_matches"] = [
            {
                "subject":     m["example"]["subject"],
                "attack_type": m["example"]["attack_type"],
                "similarity":  m["similarity"],
                "risk_score":  m["example"]["risk_score"],
            }
            for m in matches
        ]
        clean_data.append(clean_email)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)
    print(f"[Save] RAG results saved to '{path}'.")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AI Email Safety Scanner — Phase 5: RAG")
    print("=" * 55)

    # Step 1: Build/load RAG index
    rag = get_rag()

    # Step 2: Load agent results from Phase 4
    try:
        with open("agent_results.json", "r", encoding="utf-8") as f:
            emails = json.load(f)
    except FileNotFoundError:
        print("[Error] agent_results.json not found. Run agent.py first.")
        exit(1)

    # Step 3: Enrich all emails with RAG similarity
    enriched = rag.enrich_all(emails, top_k=3)

    # Step 4: Print sample results
    print("\n── RAG Similarity Results ────────────────────────────────")
    for email in enriched:
        subj    = email.get("subject", "")[:50]
        verdict = email.get("verdict", "?")
        matches = email.get("rag_top_matches", email.get("rag_matches", []))

        print(f"\n  Email   : {subj}")
        print(f"  Verdict : {verdict}")
        print(f"  RAG Context:")
        print(f"  {email.get('rag_context', 'None')}")

    # Step 5: Save for Phase 6 (LLM reasoning) and Phase 8 (dashboard)
    save_rag_results(enriched)
