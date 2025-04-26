import os
from langchain.llms import Ollama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# === 1. Load and split the text file ===
def load_and_split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(raw_text)
    return [Document(page_content=t) for t in texts]

# === 2. Create and optionally save FAISS vectorstore ===
def create_and_save_vectorstore(documents, index_path):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)
    return db

# === 3. Load FAISS vectorstore from disk ===
def load_vectorstore(index_path):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# === 4. Set up Ollama LLM ===
def setup_llm():
    return Ollama(model="gemma3:1b")

# === 5. Build RAG pipeline ===
def build_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === 6. Ask a question ===
def ask_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]

base_dir = os.getcwd()

# === MAIN ===
if __name__ == "__main__":
    
    file_path = os.path.join(base_dir, "data", "data.txt")
    index_path = os.path.join(base_dir, "vectorstorage")

    print("🔍 Preparing vector store...")
    if os.path.exists(index_path):
        print("📦 Loading existing vector store...")
        vectorstore = load_vectorstore(index_path)
    else:
        print("🧠 Creating new vector store from text...")
        documents = load_and_split_text(file_path)
        vectorstore = create_and_save_vectorstore(documents, index_path)

    print("🤖 Setting up local Ollama model...")
    llm = setup_llm()

    print("🔗 Building QA pipeline...")
    qa_chain = build_qa_chain(vectorstore, llm)

    # Interactive Q&A
    print("\n💬 Ask questions based on your file (type 'exit' to quit):")
    while True:
        question = input(">> ")
        if question.lower() in ['exit', 'quit']:
            break
        answer = ask_question(qa_chain, question)
        print(f"\n📝 Answer: {answer}\n")
