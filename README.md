# Minerva-RAG-Grammar
# 🇮🇹 Italian Grammar Voice Assistant (RAG Pipeline)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **Master's Thesis Project** > An interactive AI tutor that listens to Italian speech, analyzes grammar using Retrieval-Augmented Generation (RAG), and provides vocal feedback.

## 📖 Overview

This project implements a local Voice Assistant designed to help students learn Italian grammar. Unlike standard chatbots, this system uses **RAG (Retrieval-Augmented Generation)** to ground its answers in specific grammatical rules and examples provided by the user (e.g., textbooks or exam questions).

The system runs entirely on **Google Colab (T4 GPU)** or local machines with GPU support.

### 🎯 Key Features
* **Speech-to-Text (STT):** Uses OpenAI's **Whisper** for accurate transcription of Italian speech.
* **Dynamic RAG Knowledge Base:** Allows users to upload `.txt` files (grammar rules, A/B tests) to instantly update the AI's knowledge without retraining.
* **LLM Backend:** Powered by **Minerva-7B (SapienzaNLP)**, a Large Language Model specifically trained on Italian data.
* **Text-to-Speech (TTS):** Converts the AI's analysis back into audio for a natural conversational loop.
* **Interactive UI:** Built with **Gradio** for easy recording and file management.

## 🏗️ Architecture

The pipeline follows a modular RAG approach:

```mermaid
graph TD
    A[🎤 User Voice] -->|Whisper| B(📝 Text Transcription)
    C[📂 Grammar Rules .txt] -->|Ingestion| D{🧠 Knowledge Base}
    B --> E[Prompt Engineering]
    D -->|Context Retrieval| E
    E -->|Context + Query| F[🤖 Minerva LLM]
    F -->|Grammar Analysis| G(💬 Text Response)
    G -->|gTTS| H[🔊 Audio Feedback]

🛠️ Tech Stack
Language: Python

Interface: Gradio

Speech Recognition: OpenAI Whisper

LLM: SapienzaNLP/Minerva-7B-instruct-v1.0

Orchestration: LangChain

Audio Processing: gTTS, Librosa

🚀 How to Run
Click the Open in Colab badge above.

Connect to a T4 GPU runtime.

Install dependencies (automatically handled in the first cell).

Run the main application cell.

Click the public gradio.live link.

Usage
Upload Context: Drag and drop your grammar rule files (.txt) into the sidebar.

Update: Click "Update Knowledge Base".

Speak: Record your question or sentence in Italian.

Listen: The AI will analyze your grammar based on the uploaded rules.

📂 Project Structure
.
├── app.py               # Main application logic (Gradio + RAG pipeline)
├── requirements.txt     # List of python dependencies
├── data/                # Sample grammar rules (for testing)
│   ├── ab_questions.txt
│   └── grammar_rules.txt
└── README.md            # Project documentation
🤝 Acknowledgments
Sapienza NLP Team for the Minerva LLM.

OpenAI for the Whisper model.

Hugging Face for the transformers library.
