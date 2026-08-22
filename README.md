# Multimodal RAG-Based Intelligent Document QA System 📄

Hey there! Welcome to my B.Tech Final Year Mini Project. 

This is a Retrieval-Augmented Generation (RAG) system built to query complex documents like PDFs and get answers strictly based on the uploaded content. My main goal here was to achieve **zero hallucinations** by making sure every answer is backed by an exact source citation (page number). 

## What does it do?
- You can upload multiple PDFs (up to 25MB each).
- It extracts the text, splits it into chunks, and generates vector embeddings using Google's models.
- You can ask questions in natural language, and it streams the answers right back to you, complete with page citations so you can verify the information.

## Tech Stack
- **Frontend**: Streamlit
- **LLM**: Google Gemini (Flash)
- **Embeddings**: Google Gemini Embeddings
- **Vector Database**: Qdrant Cloud 
- **Framework**: LangChain

## How to run it locally
If you want to spin this up on your own machine:

1. Clone the repo:
   ```bash
   git clone https://github.com/official-backbencher-11/RAG-DOCUMENT-QA.git
   cd RAG-DOCUMENT-QA
   ```
2. Set up a virtual environment and install the requirements:
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   source venv/bin/activate 
   pip install -r requirements.txt
   ```
3. Set up your environment variables by renaming `.env.example` to `.env` and adding your API keys (Gemini and Qdrant).
4. Run the app:
   ```bash
   streamlit run app.py
   ```

Enjoy querying your documents!
