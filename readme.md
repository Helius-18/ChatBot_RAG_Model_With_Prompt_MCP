# 🤖 Local Chatbot with MCP Tools & RAG-Powered Memory

Welcome to your local, private, and fun-to-use AI chatbot! 🎉  
This bot is supercharged with a **local LLM (Gemma 3B)** using **Ollama**, and it can also access powerful internal tools (MCP) like fetching user data or updating emails! 🛠️  
Oh, and it remembers stuff using a RAG pipeline 🔁 thanks to your own `data.txt` file!

---

## 🧠 What This Project Does

- Loads your internal documentation (from `data/data.txt`) 📄
- Creates semantic search using FAISS + HuggingFace Embeddings 🔍
- Answers questions using Ollama + Gemma3:1b 🤯
- Can intelligently decide whether to answer from knowledge or call a tool ⚙️
- Interactive CLI mode for chatting with your data 💬

---

## 📂 Folder & File Structure

```
📁 data/
  └── data.txt            <- Your knowledge base text
📁 vectorstorage/         <- Stores FAISS index (auto-created)
📄 ChatBotWithMCP.py      <- Main script
📄 requirements.txt       <- Python dependencies
```

---

## 🧼 Resetting RAG Context (Important!)

Want to change your knowledge base?

1. Replace the content inside `data/data.txt` 📝  
2. **Delete the `vectorstorage` folder** manually 🗑️  
3. Run the script again — it will rebuild the index 🔄

---

## 🧪 Prerequisites

Make sure you have the following installed:

- ✅ [Python 3.9+](https://www.python.org/downloads/)
- ✅ [Ollama](https://ollama.com/) installed and running locally
- ✅ `gemma3:1b` model pulled via:
  ```bash
  ollama pull gemma3:1b
  ```

---

## 📦 Setup Instructions

1. 🔃 **Clone this repo** (or download it)

2. 🧪 **Set up a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate         # macOS/Linux
   venv\Scripts\activate            # Windows
   ```

3. 📦 **Install required libraries**

   ```bash
   pip install -r requirements.txt
   ```

4. 💻 **Run the chatbot**

   ```bash
   python ChatBotWithMCP.py
   ```

5. 🧠 Start asking questions — or tell it to use tools like:
   - `Get the user with ID 123`
   - `Update email for user 456 to abc@example.com`
   - Or ask anything based on your custom `data.txt`!

---

## ✨ Example Interactions

```text
>> Who is user 123
📝 Answer: User with userid 123 is bhanu

>> What does the policy say about VPN usage?
📝 Answer: [Response generated from RAG based on your data.txt]
```

---

## 🤝 Contributing

This is a personal sandbox project — but feel free to fork it, tweak it, and make it better! ❤️

---

## 🧠 Tip

The bot auto-detects if a tool should be used by understanding your intent (via LLM)! No keyword hacking needed 🔍

---

Enjoy your private AI chatbot! 🚀  