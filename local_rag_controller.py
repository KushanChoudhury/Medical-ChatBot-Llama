import requests
from flask import Flask, render_template, request, jsonify, render_template_string
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os


COLAB_ENDPOINT = "https://phrenic-patently-mac.ngrok-free.dev" 


PINECONE_API_KEY =  os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical-rag-book" 

app = Flask(__name__)



print("Initializing Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("✅ Connected to Pinecone Index.")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    print("Check your API KEY and ensure the Index Name matches exactly.")



def retrieve_documents(query, top_k=3):
   
    try:
        # Generate embedding for the query
        query_vector = embedder.encode(query).tolist()
        
        # Query Pinecone
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        docs = []
        for match in search_results['matches']:
            # Handle different metadata structures just in case
            if 'metadata' in match:
                if 'text' in match['metadata']:
                    docs.append(match['metadata']['text'])
                elif 'content' in match['metadata']:
                    docs.append(match['metadata']['content'])
                
        if not docs:
            return ["No relevant context found in the medical book index."]
            
        return docs

    except Exception as e:
        print(f"Retrieval Error: {e}")
        return []



@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('message', '')
    
    print(f"Searching for: {user_query}")
    retrieved_docs = retrieve_documents(user_query)
    context_str = "\n\n".join(retrieved_docs)
    
    system_prompt = """You are a helpful medical assistant relying on the provided context.
    - Answer ONLY based on the context provided below.
    - If the answer is missing from the context, state "I cannot find this information in the provided medical book."
    - Do not halluncinate medical advice.
    """

    user_prompt = f"""
    CONTEXT FROM MEDICAL BOOK:
    ---------------------
    {context_str}
    ---------------------

    QUESTION: 
    {user_query}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if "YOUR_COLAB" in COLAB_ENDPOINT:
        return jsonify({"response": "🚨 CONFIG ERROR: Please paste your Colab ngrok URL into 'local_rag_app.py'."})

    try:
        endpoint = COLAB_ENDPOINT.rstrip('/') + "/generate"
        
        response = requests.post(
            endpoint,
            json={
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 256
            },
            timeout=120
        )
        
        if response.status_code == 200:
            llm_result = response.json()
            bot_reply = llm_result['choices'][0]['message']['content']
            
            
            final_context = retrieved_docs
            
            if "I cannot find this information" in bot_reply:
                final_context = [] 

            return jsonify({"response": bot_reply, "context": final_context})
        else:
            return jsonify({"response": f"Error from GPU server ({response.status_code}): {response.text}"})

    except Exception as e:
        return jsonify({"response": f"Connection Error: {str(e)}"})

if __name__ == '__main__':
    print("Starting Local RAG Controller...")
    print("Go to http://localhost:8000 to chat")
    app.run(port=8000, debug=True)