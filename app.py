import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt
from langchain_groq import ChatGroq

app = Flask(__name__)
load_dotenv()

# Global state for lazy initialization
conversational_rag_chain = None
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Lazy initializer to prevent blocking Gunicorn startup and container health checks."""
    global conversational_rag_chain
    if conversational_rag_chain is None:
        embeddings = download_hugging_face_embeddings()
        index_name = "medical-chatbot"

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name, embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 2})

        llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.5
        )

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

        conversational_rag_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    return conversational_rag_chain

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg") or request.form.get("message")
    if not user_input and request.is_json:
        user_input = request.json.get("msg")

    if not user_input:
        return "No input provided", 400

    try:
        chain = get_rag_chain()
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default_user_session"}}
        )
        return str(response)
    except Exception as e:
        print(f"Error during execution: {e}")
        return f"Backend Error: {str(e)}", 500

@app.route("/clear", methods=["POST"])
def clear_history():
    if "default_user_session" in store:
        store["default_user_session"].clear()
    return jsonify({"status": "success", "message": "Memory cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)