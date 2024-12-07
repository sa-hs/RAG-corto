# RAG-corto
Este repositorio contiene un código diseñado para implementar un sistema de RAG utilizando herramientas avanzadas de la biblioteca LangChain y sus extensiones comunitarias.

import subprocess
# Configurar el entorno e instalar paquetes
try:
    subprocess.run(['pip', 'install', 'langchain', 'langchain_community', 'langchain-openai',
                    'scikit-learn', 'langchain-ollama', 'pymupdf', 'langchain_huggingface',
                    'faiss-gpu'], check=True)
    print("Paquetes instalados correctamente.")
except subprocess.CalledProcessError as e:
    print("Error al instalar paquetes:", e)

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import  PyMuPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader =  PyMuPDFLoader('/content/drive/MyDrive/Colab Notebooks/CIDE/Introducción a la Estadística.pdf')
docs = loader.load()

!curl -fsSL https://ollama.com/install.sh | sh

!pip install colab-xterm
%load_ext colabxterm
%xterm

#ollama serve
!ollama pull llama3

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)

llm = Ollama(model="llama3")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)

question = "que es un estadistico?"
result = chain.invoke({"query": question})

print(result['result'])
