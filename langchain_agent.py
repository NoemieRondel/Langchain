import os
import pandas as pd
import numpy as np
import faiss
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Variables d'environnement pour Azure OpenAI
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY_4')
azure_openai_api_endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT_4')
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME_4")

# Initialiser le modèle d'embedding et le modèle de chat Azure OpenAI
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    azure_deployment='text-embedding-3-large',
    azure_endpoint=azure_openai_api_endpoint,
    openai_api_version="2023-05-15",
    chunk_size=500
)

model = AzureChatOpenAI(api_key=azure_openai_api_key,
                        api_version="2023-12-01-preview",
                        azure_endpoint=azure_openai_api_endpoint,
                        model=deployment_name,
                        temperature=0
                        )

# Chemin pour enregistrer l'index FAISS
faiss_index_file = "embeddings_index.faiss"
embedding_dim = 3072  # Vérifie si cette dimension est correcte pour ton modèle

# Fonction pour charger les documents depuis un fichier CSV
def load_documents(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        documents = [
            Document(page_content=row['summary'], metadata={"title": row['title'], "author": row['author']})
            for _, row in df.iterrows()
        ]
    else:
        raise ValueError("Format de fichier non supporté. Veuillez utiliser un fichier CSV.")
    
    return documents

# Fonction pour créer un nouvel index FAISS et l'associer aux documents
def create_index_faiss(embeddings, documents, save_path):
    index = faiss.IndexFlatL2(embedding_dim)

    embeddings_array = np.array(embeddings, dtype='float32')
    index.add(embeddings_array)

    # Création du Docstore en mémoire et mapping index-docstore
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}  # Assure-toi que les IDs soient des chaînes

    print("DEBUG: index_to_docstore_id", index_to_docstore_id)

    # Création du FAISS store avec les arguments corrects
    faiss_store = FAISS(index, docstore, index_to_docstore_id)

    # Sauvegarde de l'index FAISS
    faiss.write_index(index, save_path)

    return faiss_store

# Fonction pour charger l'index FAISS
def load_index_faiss(save_path, documents):
    if os.path.exists(save_path):
        index = faiss.read_index(save_path)

        # Création du Docstore en mémoire et mapping index-docstore
        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        print("DEBUG: index_to_docstore_id (load)", index_to_docstore_id)

        # Charger l'index FAISS avec docstore et index_to_docstore_id
        faiss_store = FAISS(index, docstore, index_to_docstore_id)
        return faiss_store
    else:
        return None

# Fonction pour calculer les embeddings des documents
def calculate_embeddings(documents):
    return [embedding_model.embed_query(doc.page_content) for doc in documents]

# Charger les documents à partir d'un fichier CSV
root_books = "cleaned_books.csv"
documents = load_documents(root_books)

# Charger ou créer un index FAISS
faiss_index = load_index_faiss(faiss_index_file, documents)

if faiss_index is None:
    print("Index FAISS introuvable, génération des embeddings et création d'un nouvel index...")
    embeddings = calculate_embeddings(documents)
    faiss_index = create_index_faiss(embeddings, documents, faiss_index_file)
else:
    print("Index FAISS chargé depuis le fichier.")

# Fonction pour créer l'agent LangChain
def create_agent():
    qa_prompt_template = PromptTemplate(
        input_variables=["question", "documents"],
        template="""Répondez à la question suivante '{question}' en vous basant sur les documents suivants : {documents}"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=faiss_index.as_retriever()
    )
    return qa_chain

# Fonction pour poser une question à l'agent
def ask_question(question):
    agent = create_agent()
    response = agent.invoke(question)
    return response

# Exemple d'appel de la fonction poser_question
if __name__ == "__main__":
    question = "Résume-moi l'oeuvre de Balzac?"
    reponse = ask_question(question)
    print("Réponse :", reponse)
