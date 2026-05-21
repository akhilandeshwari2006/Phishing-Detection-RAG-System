# AI Email Safety Scanner (Phishing Detection RAG System)

A Python-based system that scans your Gmail inbox for phishing emails using a combination of machine learning, retrieval-augmented generation (RAG), and an LLM. Results are displayed in a Streamlit dashboard.

---

## Features

- Real-time phishing email detection
- Gmail inbox monitoring
- ML-based classification
- Retrieval-Augmented Generation (RAG)
- AI-generated threat explanations
- Streamlit dashboard visualization

---

## How it works

Emails are fetched from Gmail via the API, cleaned and preprocessed, then passed through a trained ML classifier. Suspicious emails are further analyzed using a FAISS-based RAG retrieval step and an LLM (OpenAI or Anthropic) to generate a human-readable threat explanation. Alerts can be sent automatically, and everything is viewable in the dashboard.

```
Gmail API → Preprocessing → ML Detection → FAISS Retrieval → LLM Analysis → Dashboard
```

---

## Tech Stack

- Python
- Scikit-learn
- FAISS
- LangChain
- OpenAI / Anthropic APIs
- Streamlit
- Gmail API

---

## Project structure

```
├── app.py               # Streamlit dashboard
├── agent.py             # LangChain pipeline orchestration
├── gmail_fetcher.py     # Fetches emails via Gmail API
├── preprocess.py        # Cleans and extracts email text
├── detector.py          # ML phishing classifier
├── rag.py               # FAISS-based RAG retrieval
├── llm.py               # LLM integration
├── notify.py            # Sends alerts for flagged emails
├── output.py            # Formats results
├── auto_scan.py         # One-time inbox scan
├── watch_inbox.py       # Continuous inbox monitoring
├── auth_fix.py          # OAuth helper
├── phishing_model.pkl   # Pre-trained scikit-learn model
├── rag_index.faiss      # FAISS vector index
├── rag_vectorizer.pkl   # Vectorizer for RAG
├── rag_examples.json    # Example phishing cases
├── requirements.txt
├── env.example
└── SETUP.md             # Gmail API setup guide
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/akhilandeshwari2006/Phishing-Detection-RAG-System.git
cd Phishing-Detection-RAG-System

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp env.example .env
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 3. Gmail API credentials

See [SETUP.md](SETUP.md) for the full walkthrough. In short:

- Create a Google Cloud project and enable the Gmail API
- Set up an OAuth 2.0 Desktop App credential
- Download and save it as `credentials.json` in the project root

### 4. Authenticate

```bash
python gmail_fetcher.py
```

A browser window will open for Gmail login. After that, `token.json` is saved and you won't need to log in again.

---

## Usage

| Command | What it does |
|---|---|
| `python gmail_fetcher.py` | Fetch emails and save to `emails.json` |
| `python auto_scan.py` | One-time scan of your inbox |
| `python watch_inbox.py` | Continuously monitor for new emails |
| `streamlit run app.py` | Open the dashboard at `localhost:8501` |

---

## Screenshots

### Phishing Email Detection
![Phishing Detection](Screenshot%202026-05-21%20174809.png)

### Safe Email Detection
![Safe Email Detection](Screenshot%202026-05-21%20174745.png)

## Notes

- `credentials.json` and `token.json` are gitignored — never commit them
- The app works with either OpenAI or Anthropic; set whichever key you have in `.env`
