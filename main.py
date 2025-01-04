from flask import Flask, request, render_template, jsonify
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

app = Flask(__name__)

# Initialize global variables
vectorstore = None
api_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = api_key
VECTORSTORE_PATH = "vectorstore"

# LLM Configuration
repo_id = "microsoft/Phi-3.5-mini-instruct"
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

# Load and Split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    return chunks

# Embed and Store Documents
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VECTORSTORE_PATH)
    vectorstore.persist()  # Save the vector store to disk
    return vectorstore

# Load Vector Store if Exists
def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading vector store from disk...")
        vectorstore = Chroma(persist_directory=VECTORSTORE_PATH)
        return vectorstore
    print("No vector store found. Please upload a PDF first.")
    return None

# Configure LLM and Prompt
def configure_llm():
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        huggingfacehub_api_token=api_key
    )
    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:

    {context}

    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    return prompt, llm

# Run RAG Chain
def run_rag_chain(question, vectorstore, prompt, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    llm_chain = prompt | llm | StrOutputParser()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    response = rag_chain.invoke(question)
    response = response.replace("</s>", "").strip()
    return response

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global vectorstore  # Ensure global variable usage
    file = request.files['pdf']
    
    if file:
        try:
            # Save uploaded PDF
            pdf_path = os.path.join("uploads", file.filename)
            file.save(pdf_path)

            # Load and split PDF into chunks
            chunks = load_and_split_pdf(pdf_path)
            
            # Create vectorstore from chunks
            vectorstore = create_vectorstore(chunks)

            return jsonify({"message": "PDF uploaded and processed successfully. You can now ask questions."})

        except Exception as e:
            return jsonify({"error": f"Error processing the PDF: {str(e)}"}), 500

    return jsonify({"error": "Failed to upload PDF. Ensure a valid PDF file is selected."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore  # Ensure global variable usage
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    if vectorstore is None:
        return jsonify({"error": "No document loaded. Please upload a PDF first."}), 400

    try:
        prompt, llm = configure_llm()
        response = run_rag_chain(question, vectorstore, prompt, llm)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": f"Error processing the question: {str(e)}"}), 500

if __name__ == "__main__":
    vectorstore = load_vectorstore()  # Try to load the vector store from disk
    os.makedirs("uploads", exist_ok=True)  # Ensure the 'uploads' folder exists
    app.run(debug=True)
