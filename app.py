from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key= os.getenv("OPENAI_API_KEY"),
    model = "text-embedding-3-large"
)

index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

prompt = """Using the given {context}, provide a detailed and accurate answer to the user's question in the question's language. 
Make sure to rely only on the information provided in the context without introducing any external knowledge or 
assumptions. Here's the user's question: {question}."""

prompt_template = PromptTemplate(template=prompt)

chain = LLMChain(llm=llm, prompt=prompt_template)

def get_response(question):
    docs = vector_store.similarity_search(question)
    response = chain.run(context=docs, question=question)
    return str(response)


# print(get_response("What is the best diploma for someone with background in Python"))

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello, World!</h1>"

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "success",
            "message": "Service is up and running",
        }
    )
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Invalid request"}), 400
    try:
        query = data["query"]
        response = get_response(query)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)