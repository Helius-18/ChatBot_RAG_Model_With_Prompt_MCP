import os
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.tools import Tool
import json

# === 1. Load and split the text file ===
def load_and_split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(raw_text)
    return [Document(page_content=t) for t in texts]

# === 2. Create and optionally save FAISS vectorstore ===
def create_and_save_vectorstore(documents, index_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)
    return db

# === 3. Load FAISS vectorstore from disk ===
def load_vectorstore(index_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# === 4. Set up Ollama LLM ===
def setup_llm():
    return Ollama(model="gemma3:1b")

# === 5. Build RAG pipeline ===
def build_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def get_user_by_id(userId: int):
    return f"user with userid {userId} is bhanu"

def update_user_email(userId: int, email : str):
    return f"user with userid {userId} is {email}"

tools = [
    Tool(
        name="get_user_by_id",
        func=lambda x: get_user_by_id(int(json.loads(x)["userId"])),
        description="Fetch user details by ID. Input: {'userId': <integer>}"
    ),
    Tool(
        name="update_user_email",
        func=lambda x: update_user_email(int(json.loads(x)["userId"]), json.loads(x)["email"]),
        description="Update a user's email. Input: {'userId': <integer>, 'email': <string>}"
    )
]

# === 6. Ask a question ===
def ask_question(qa_chain, question, llm):

    intent_prompt = f"""
    You have access to MCP tools: {tools}.
    Based on the user's input: "{question}", decide if an MCP tool should be called.
    Action in the json should "tool" when its the MCP tool not the toolname.
    If you find the tool, check in the question if we have all the parameters if not just switch the action to direct.

    Example 1:
    User input: "Hi, I just need help"
    Response:
    {{"action": "direct", "response": "Do not respond anything"}}
    
    Example 2:
    User input: "Get user with ID 123"
    Response:
    {{"action": "tool", "tool": "get_user_by_id", "params": {{"userId": 123}}}}

    Now, respond to this input in the same format (no code blocks, no markdown):
    "{question}"
    """

    response_raw = llm.invoke(intent_prompt)

    response_raw = response_raw.strip()

    # Remove triple backticks (```), if they exist at the start and end
    if response_raw.startswith("```json") and response_raw.endswith("```"):
        response_raw = response_raw[7:-3].strip()

    # Check if the response is empty or not a valid JSON string
    if not response_raw:
        print("Empty response")
        return "No valid response from model"

    try:
        response = json.loads(response_raw)
        
        if response.get('action') == 'direct':
            result = qa_chain.invoke({"query": question})
        elif response.get('action') == 'tool':
            tool_name = response.get('tool')
            params = response.get('params')

            for tool in tools:
                if tool.name == tool_name:
                    try:
                        tool_result = tool.func(json.dumps(params))
                        result = llm.invoke(f"Rephrase this to a single-line response: \"{tool_result}\". No explanations, just the sentence.")
                        return result
                    except (ValueError, KeyError) as e:
                        return "Sorry, I cant help you with that"

            return "Tool not found"
        else:
            return "Response does not match expected action"

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return "Error: Invalid JSON response"

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
        answer = ask_question(qa_chain, question, llm)
        print(f"\n📝 Answer: {answer}\n")
