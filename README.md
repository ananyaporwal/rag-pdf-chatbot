# RAG Chatbot — Day 1 Setup

## What you're building
A Python chatbot that reads any PDF and answers questions about it.

---

## Setup (do this once)

### 1. Make sure Python is installed
```
python --version
```
Should show Python 3.9 or higher.

### 2. Create a project folder
```
mkdir rag_chatbot
cd rag_chatbot
```

### 3. Create a virtual environment (keeps libraries organized)
```
python -m venv venv
```

Activate it:
- Windows:   venv\Scripts\activate
- Mac/Linux: source venv/bin/activate

You'll see (venv) appear in your terminal. Good.

### 4. Install all libraries
```
pip install -r requirements.txt
```
This takes 2-3 minutes. Let it run.

### 5. Get your OpenAI API key (FREE $5 credit on signup)
- Go to: https://platform.openai.com/api-keys
- Sign up / Log in
- Click "Create new secret key"
- Copy the key (starts with sk-)

### 6. Paste key into .env file
Open .env and replace: your_openai_api_key_here
With your actual key: sk-xxxxxxxxxxxxxxxx

### 7. Add a PDF file to the folder
- Copy any PDF into your rag_chatbot folder
- Rename it to sample.pdf
- OR change PDF_PATH in day1_rag.py to your filename

---

## Run it
```
python day1_rag.py
```

---

## Expected output
```
Reading PDF: sample.pdf
Total characters extracted: 24500
Splitting text into chunks...
Total chunks created: 52
Creating embeddings and storing in FAISS...
Vector store created successfully!

==================================================
PDF loaded! You can now ask questions.
Type 'quit' to exit.
==================================================

Your question: What is this document about?

Searching document...

Answer: This document is about...
Sources used:
  [1] ...relevant text from PDF...
```

---

## Common errors and fixes

**Error: ModuleNotFoundError**
Fix: Make sure your venv is activated and run pip install -r requirements.txt again

**Error: openai.AuthenticationError**
Fix: Check your API key in .env — make sure there are no spaces

**Error: File not found**
Fix: Make sure your PDF is in the same folder as day1_rag.py

---

## Day 2 preview
Tomorrow we add the Streamlit UI — turning this terminal script
into a real web app with file upload and chat interface.
