import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt

app = Flask(__name__)
load_dotenv()

# 1. Initialize Vector Store & Embeddings
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# 2. Initialize LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

# Helper to format retrieved Pinecone docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Build Base LCEL Chain
# The chain expects 'input', 'chat_history', and 'context'
base_chain = (
    {
        "context": (lambda x: x["input"]) | retriever | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Memory Store Management
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with automatic message history tracking
conversational_rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg") or request.form.get("message")

    if not user_input:
        return "No input provided", 400

    print(f"User Question: {user_input}")

    # Pass a session_id so the memory store tracks history for this user session
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default_user_session"}}
    )

    print(f"Bot Response: {response}")
    return str(response)

@app.route("/clear", methods=["POST"])
def clear_history():
    if "default_user_session" in store:
        store["default_user_session"].clear()
    return jsonify({"status": "success", "message": "Memory cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)