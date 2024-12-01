import os
os.environ["OPENAI_API_KEY"] = "Your API Key"

from langchain.embeddings import OpenAIEmbeddings
import chromadb
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

persistent_client = chromadb.PersistentClient()

def load_documents(file_path):
    """
    Load documents from a PDF or Word file.
    Automatically detects the file type based on the extension.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please use a PDF or DOCX file.")
    
    documents = loader.load()
    return documents
def create_vector_store(documents):
    """
    Generate vector embeddings and create a FAISS index.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    for i,data in enumerate(all_splits):
        embeddings.embed_query(all_splits[i].page_content)
    vector_store = Chroma(
    client=persistent_client,
    collection_name="abc",
    embedding_function=embeddings)
    return vector_store
def create_chat_chain(vector_store):
    """
    Create a conversational retrieval chain.
    """
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.9)
    chat_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    return chat_chain
def run_chatbot(chat_chain):
    """
    Interact with the chatbot in a conversational loop.
    """
    chat_history = []
    print("Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = chat_chain({"question": query, "chat_history": chat_history})
        print(f"Bot: {response['answer']}")
        chat_history.append((query, response['answer']))
file_path = r"sample.pdf"
documents = load_documents(file_path)
vector_store = create_vector_store(documents)
chat_chain = create_chat_chain(vector_store)
run_chatbot(chat_chain)  
