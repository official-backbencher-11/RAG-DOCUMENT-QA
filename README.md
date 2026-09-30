# Multimodal RAG-Based Intelligent Document QA System 📄

A Retrieval-Augmented Generation (RAG) app for asking questions about PDF documents. Answers come **only** from the uploaded content and always include page citations (file name + page).

## Features
- Upload multiple PDFs (up to 25 MB each); text is chunked (1000 chars, 150 overlap) and embedded with Google Gemini.
- Vectors are stored in Qdrant Cloud; the top 4 matching chunks are sent to Gemini, which streams the answer.
- Each browser session only searches its own documents (no cross-user leakage); re-uploading a file does not create duplicates.
- Says "I cannot find this information in the uploaded document." when the answer is not in the PDFs.
- If Qdrant Cloud is unreachable the app shows the reason and falls back to temporary in-memory storage.

## Tech stack
Streamlit · LangChain · Google Gemini (LLM + embeddings) · Qdrant

## Run locally
```bash
git clone https://github.com/official-backbencher-11/RAG-DOCUMENT-QA.git
cd RAG-DOCUMENT-QA
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
streamlit run app.py
```

## Deploy on Streamlit Cloud
Add the keys under *Manage app → Settings → Secrets* (see `.streamlit/secrets.toml.example`):
`GEMINI_API_KEY`, `QDRANT_URL` (e.g. `https://xxxx.region.cloud.qdrant.io:6333`), `QDRANT_API_KEY`.
Optional: `GEMINI_MODEL`, `EMBEDDING_MODEL` to override the default models.

**"Connection reset by peer" / cannot connect to Qdrant?** Free Qdrant Cloud clusters are suspended after inactivity. Open cloud.qdrant.io, resume (or recreate) the cluster and copy its URL and API key again.

## Tests
```bash
pip install pytest reportlab
pytest
```
