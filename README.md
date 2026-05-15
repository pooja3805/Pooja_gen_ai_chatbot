# College FAQ Chatbot

A simple AI-powered College FAQ Chatbot built using Python, Transformers, Sentence Transformers, and FAISS.

## Features
- Answers college-related questions using Retrieval-Augmented Generation (RAG)
- Semantic search with embeddings
- Handles FAQ queries like admissions, fees, hostel, placements, courses, etc.
- Easy to customize by editing `college_data.txt`

## Tech Stack
- Python
- Transformers
- Sentence Transformers
- FAISS
- Google Colab / Jupyter Notebook

## Project Structure
```bash
college_chatbot/
│
├── chatbot.py
├── college_data.txt
├── README.md
```

## Installation

Install dependencies:

```bash
pip install transformers sentence-transformers faiss-cpu
```

## Run Project

```bash
python chatbot.py
```

## Sample Questions
- What is the admission criteria?
- What are hostel fees?
- Is WiFi available?
- What courses are offered?
- What is the attendance requirement?

## Dataset
The chatbot uses `college_data.txt` as its knowledge base.

Example topics:
- Admissions
- Fees
- Hostel
- Scholarships
- Library
- Placements
- Courses
- Transport
- Sports
- Medical facilities

## Future Improvements
- PDF support
- Web interface using Streamlit
- Voice chatbot
- Multi-language support

## Author
Manasvini
