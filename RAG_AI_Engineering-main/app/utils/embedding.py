import argparse  # Importe le module argparse pour analyser les arguments de la ligne de commande
import os  # Importe le module os pour interagir avec le système d'exploitation
import pickle  # Importe le module pickle pour la sérialisation et la désérialisation d'objets Python
import random  # Importe le module random pour générer des nombres aléatoires
import tempfile  # Importe le module tempfile pour créer des fichiers temporaires

# Importation des chargeurs de documents de Langchain
from langchain.document_loaders import (CSVLoader, PyPDFDirectoryLoader,
                                        PyPDFLoader, TextLoader)
from langchain.document_loaders.csv_loader import \
    CSVLoader  # Importe le chargeur de documents CSV
# Importation des embeddings de HuggingFace de Langchain
from langchain.embeddings import \
    HuggingFaceEmbeddings  # HuggingFaceInstructEmbeddings
# Importation des récupérateurs de Langchain
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# Importation des diviseurs de texte de Langchain
from langchain.text_splitter import CharacterTextSplitter  # recursive
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importation des magasins de vecteurs de Langchain
from langchain.vectorstores import FAISS

# Fonction de création de base de données vectorielle (commentée)
#def vector_db_creation ( file ) :
    ##filename = './temp/' + str(random.randint(10**6, 10**7))+'.pdf'
    #with open(file.name, mode='wb') as w:
        #w.write(file.getvalue())

    #loader = PyPDFLoader(file.name)
    #textsplitter = RecursiveCharacterTextSplitter(
        ##separator="\n",
        #chunk_size=1000,
        #chunk_overlap=100
    #)
    #documentchunks = loader.load_and_split(text_splitter=textsplitter)

############################

def load_file(file):
    # Sauvegarde du fichier téléchargé dans un emplacement temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getbuffer())
        temp_file_path = tmp_file.name
    
    file_extension = os.path.splitext(file.name)[1].lower()  # Obtenir l'extension du fichier en minuscules
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file_path)  # Charger un fichier PDF
    elif file_extension == '.csv':
        loader = CSVLoader(temp_file_path)  # Charger un fichier CSV
    elif file_extension == '.txt':
        loader = TextLoader(temp_file_path)  # Charger un fichier texte
    else:
        raise ValueError("Unsupported file type: " + file_extension)  # Lever une erreur si le type de fichier n'est pas supporté
    
    return loader, temp_file_path  # Retourner le chargeur et le chemin du fichier temporaire

def vector_db_creation(file):
    loader, temp_file_path = load_file(file)  # Charger le fichier et obtenir le chemin du fichier temporaire
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Taille des morceaux plus petite pour une récupération plus précise
        chunk_overlap=50  # Chevauchement des morceaux
    )
    documentchunks = loader.load_and_split(text_splitter=text_splitter)  # Charger et diviser le document en morceaux

    # Définir le modèle d'embeddings
    embedmodel="sentence-transformers/msmarco-distilbert-cos-v5"
    #embedmodel="hkunlp/instructor-large" # https://huggingface.co/hkunlp # 1.26GB
    #embedmodel="intfloat/multilingual-e5-base" # https://huggingface.co/intfloat/multilingual-e5-base # 1.05GB
    #embedmodel='asafaya/bert-base-arabic' ### arabic
    #embedmodel="setu4993/LEALLA-small" # https://huggingface.co/setu4993/LEALLA-small # 280Mo
    #embedmodel="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #     https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 # 480Mo

    #embeddings = HuggingFaceInstructEmbeddings(model_name=embedmodel, model_kwargs={"device": "cpu"})
    embeddings = HuggingFaceEmbeddings(model_name=embedmodel, model_kwargs={"device": "cpu"})  # Créer des embeddings avec HuggingFace

    vector_store = FAISS.from_documents(documentchunks, embeddings)  # Créer un magasin de vecteurs FAISS à partir des morceaux de documents

    retriever_lancedb = vector_store.as_retriever(search_kwargs={"k": 3})  # Créer un récupérateur de vecteurs
    bm25_retriever = BM25Retriever.from_documents(documentchunks)  # Créer un récupérateur BM25 à partir des morceaux de documents
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_lancedb],weights=[0.8, 0.2])  # Créer un récupérateur d'ensemble avec des poids spécifiques
    
    # Nettoyer le fichier temporaire
    #os.remove(temp_file_path)
    return vector_store, ensemble_retriever, temp_file_path  # Retourner le magasin de vecteurs, le récupérateur d'ensemble et le chemin du fichier temporaire
