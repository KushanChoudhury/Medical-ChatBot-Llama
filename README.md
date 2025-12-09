# Medical ChatBot (End-to-End RAG with Llama 3.1)

An end-to-end medical chatbot that uses **Retrieval Augmented Generation (RAG)** to answer questions based on a specific medical textbook. The system utilizes a hybrid architecture where the heavy LLM inference runs on a GPU (via Google Colab) while the application logic and RAG retrieval run locally.

## 🚀 Features

* **RAG Architecture**: Retrieves relevant context from a medical PDF book before answering.
* **Vector Database**: Uses **Pinecone** to store and search text embeddings.
* **LLM Power**: Powered by **Llama-3.1-8B-Instruct** (Quantized GGUF) for high-quality responses.
* **Hybrid Setup**: 
    * **Local**: Flask app for UI and Pinecone retrieval.
    * **Remote (GPU)**: Google Colab hosting the Llama model via `ngrok` tunneling.
* **Interactive UI**: Clean HTML/JS chat interface.

## 🛠️ Tech Stack

* **LLM Framework**: `llama-cpp-python`, `huggingface_hub`
* **Vector DB**: `pinecone-client`
* **Embeddings**: `sentence-transformers` (Model: `all-MiniLM-L6-v2`)
* **Backend**: Flask, Python
* **Document Processing**: `langchain`, `pypdf`
* **Tunneling**: `pyngrok`

---

## 📂 Project Structure

```text
.
├── data_ingest.py            # Script to parse PDF and upsert embeddings to Pinecone
├── local_rag_controller.py   # Main Flask app (Frontend + RAG Logic)
├── requirements.txt          # Python dependencies
├── Medical_book.pdf          # (Place your PDF here)
├── .env                      # Environment variables (API Keys)
├── templates/
│   └── index.html            # Chatbot User Interface
└── model_server/
    └── Medical0_Bot_LLM.ipynb # Google Colab Notebook for hosting the LLM
```

## ⚙️ Prerequisites
Pinecone API Key: Sign up at [Pinecone](https://www.pinecone.io/) and create an API Key.

Ngrok Auth Token: Sign up at [ngrok](https://ngrok.com/) to get an auth token (for the Colab tunnel).

Python 3.10+ installed locally.

## 📥 Installation & Setup
1. Local Environment Setup
Clone the repository and install dependencies:

```text
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

Create a .env file in the root directory:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

2. Knowledge Base Ingestion
Place your medical PDF file in the root directory and name it Medical_book.pdf. Run the ingestion script to process the book and store vectors in Pinecone:
```
python data_ingest.py
```

This will create an index named medical-rag-book in Pinecone and upload the embeddings.

## 🚀 Usage Guide
This project requires two components running simultaneously: the Model Server (Colab) and the Local Controller.

Step 1: Start the Model Server (Google Colab)
Since Llama 3.1 requires a GPU, we run it on Google Colab.

Open model_server/Medical0_Bot_LLM.ipynb in Google Colab.

Add your NGROK_AUTH_TOKEN to the Colab Secrets (key icon on the left) with the name NGROK_AUTH_TOKEN.

Run all cells.

At the end of the execution, the script will print a public URL (e.g., https://xyz-123.ngrok-free.app). Copy this URL.

Step 2: Configure Local Controller
Open local_rag_controller.py and paste the ngrok URL you copied into the COLAB_ENDPOINT variable:

```
COLAB_ENDPOINT = "[https://your-copied-url.ngrok-free.app](https://your-copied-url.ngrok-free.app)"
```

Step 3: Run the Chatbot

Start the local Flask application:

```
python local_rag_controller.py
Open your browser and navigate to: http://localhost:8000
```

## 💡 How It Works
1.User Input: You ask a question in the web UI.

2.Retrieval: local_rag_controller.py converts the question into a vector and searches Pinecone for the most relevant passages from Medical_book.pdf.

3.Prompt Construction: The retrieved passages and the user's question are formatted into a prompt.

4.Inference: This prompt is sent via HTTP (requests) to the Colab instance running the Llama 3.1 model.

5.Response: The LLM generates a medical answer based only on the provided context, which is sent back to the UI.
