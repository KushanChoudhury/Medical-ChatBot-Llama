import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv() 
# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical-rag-book"
PDF_FILE_PATH = "Medical_book.pdf"  # Put your book file name here

# Vector Dimension for 'all-MiniLM-L6-v2' is 384
VECTOR_DIMENSION = 384 

def main():
    # 1. Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Change this if your Pinecone region is different
            )
        )
        # Wait a moment for index to be ready
        time.sleep(10)
    
    index = pc.Index(PINECONE_INDEX_NAME)

    # 2. Load and Process PDF
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File '{PDF_FILE_PATH}' not found. Please place your PDF in this folder.")
        return

    print(f"Loading '{PDF_FILE_PATH}'...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 3. Split Text into Chunks
    # We split by 1000 characters with some overlap to maintain context
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Initialize Embedding Model
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # 5. Embed and Upsert to Pinecone
    print("Upserting vectors to Pinecone (this may take a while)...")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        
        # Prepare data for this batch
        ids = [f"chunk_{i + j}" for j in range(len(batch_chunks))]
        texts = [chunk.page_content for chunk in batch_chunks]
        
        # Create Embeddings
        embeddings = embedder.encode(texts).tolist()
        
        # Prepare metadata (store the actual text so we can retrieve it later)
        metadata = [
            {"text": text, "page": chunk.metadata.get("page", 0)} 
            for text, chunk in zip(texts, batch_chunks)
        ]
        
        # Zip together into Pinecone format: (id, vector, metadata)
        to_upsert = list(zip(ids, embeddings, metadata))
        
        # Upload
        index.upsert(vectors=to_upsert)
        print(f"Upserted batch {i} to {i + len(batch_chunks)}")

    print("\n✅ Success! Knowledge base ingestion complete.")

if __name__ == "__main__":
    main()