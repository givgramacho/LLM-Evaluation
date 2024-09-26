import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


load_dotenv()
vectordb_path = "Vector_db"
resumes_path = "docs"
huggingface_api = os.getenv("hugging_api_key")

docs_loader = DirectoryLoader("Documents")
documents = docs_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents= docs,
                    embedding= embeddings,
                    persist_directory= vectordb_path)

print("Documents stored in the database")