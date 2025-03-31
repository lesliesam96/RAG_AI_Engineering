import base64  # Importe le module base64 pour l'encodage et le décodage en base64
import io  # Importe le module io pour la gestion des flux de données (input/output)
import os  # Importe le module os pour interagir avec le système d'exploitation
import re  # Importe le module re pour travailler avec les expressions régulières

import fitz  # Importe la bibliothèque PyMuPDF pour travailler avec des fichiers PDF
import matplotlib.pyplot as plt  # Importe la bibliothèque matplotlib pour créer des graphiques
import numpy as np  # Importe la bibliothèque numpy pour manipuler des tableaux et effectuer des calculs mathématiques
import pandas as pd  # Importe la bibliothèque pandas pour l'analyse et la manipulation de données
import seaborn as sns  # Importe la bibliothèque seaborn pour la visualisation de données statistiques
import streamlit as st  # Importe la bibliothèque streamlit pour créer des applications web interactives
from dotenv import (  # Importe les fonctions find_dotenv et load_dotenv pour charger les variables d'environnement depuis un fichier .env
    find_dotenv, load_dotenv)
from genai import (  # Importe les classes Client et Credentials du module genai pour interagir avec l'API GenAI
    Client, Credentials)
from genai.extensions.langchain import \
    LangChainInterface  # Importe l'interface LangChainInterface de l'extension langchain du module genai
from genai.text.generation import (  # Importe diverses classes et méthodes pour la génération de texte avec genai
    DecodingMethod, ModerationHAP, ModerationParameters,
    TextGenerationParameters, TextGenerationReturnOptions)
from langchain.agents import \
    AgentExecutor  # Importe la classe AgentExecutor du module langchain
from langchain.agents.agent_types import \
    AgentType  # Importe la classe AgentType du module langchain
from langchain.chains import (  # Importe les chaînes de traitement ConversationalRetrievalChain et RetrievalQA du module langchain
    ConversationalRetrievalChain, RetrievalQA)
from langchain.document_loaders.csv_loader import \
    CSVLoader  # Importe le chargeur de documents CSVLoader pour les fichiers CSV du module langchain
from langchain.embeddings import \
    HuggingFaceEmbeddings  # Importe la classe HuggingFaceEmbeddings pour les embeddings du module langchain
from langchain.llms import \
    CTransformers  # Importe la classe CTransformers pour les modèles de langage du module langchain
from langchain.memory import \
    ConversationBufferMemory  # Importe la classe ConversationBufferMemory pour gérer la mémoire des conversations du module langchain
from langchain.prompts import \
    PromptTemplate  # Importe la classe PromptTemplate pour les modèles de questions du module langchain
from langchain.text_splitter import \
    RecursiveCharacterTextSplitter  # Importe la classe RecursiveCharacterTextSplitter pour diviser le texte en parties du module langchain
from langchain.vectorstores import \
    FAISS  # Importe la classe FAISS pour stocker et rechercher des vecteurs du module langchain
from langchain_experimental.agents import \
    create_pandas_dataframe_agent  # Importe la fonction create_pandas_dataframe_agent pour créer un agent de dataframe pandas
from langchain_experimental.agents.agent_toolkits import (  # Importe les fonctions create_csv_agent et create_pandas_dataframe_agent pour créer des agents pour les fichiers CSV et les dataframes pandas
    create_csv_agent, create_pandas_dataframe_agent)
from PIL import \
    Image  # Importe la classe Image de la bibliothèque PIL (Pillow) pour manipuler les images
from utils.embedding import vector_db_creation

DB_FAISS_PATH = "vectorstore/db_faiss"  # Définit le chemin où les données FAISS seront stockées

# Liste des modèles de langage disponibles
llms = ['ibm/granite-20b-code-instruct', 'ibm/granite-20b-multilang-lab-rc', 'ibm/granite-13b-chat-v2', 'ibm/granite-34b-code-instruct', 'ibm/granite-3b-code-instruct', 'ibm/granite-7b-lab', 'ibm/granite-8b-code-instruct', 'ibm/granite-8b-japanese-lab-rc', 'meta-llama/llama-2-13b-chat', 'ibm-meta/llama-2-70b-chat-q', 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'ibm-mistralai/merlinite-7b', 'mistralai/mixtral-8x7b-instruct-v01', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2', 'kaist-ai/prometheus-8x7b-v2']
#llms = ['google/flan-t5-xxl', 'google/flan-ul2', 'codellama/codellama-34b-instruct-hf', 'eleutherai/gpt-neox-20b', 'ibm/granite-20b-code-instruct-v1', 'ibm/granite-13b-chat-v2', 'ibm/granite-13b-instruct-v1', 'ibm/granite-13b-instruct-v2', 'google/flan-t5-xl', 'ibm/granite-13b-chat-v2', 'ibm/granite-13b-instruct-v2', 'ibm/granite-7b-lab', 'meta-llama/llama-2-13b-chat', 'google/flan-ul2']

# Configurer la page Streamlit
im = Image.open("app/images/sricon.png")  # Ouvrir une image pour l'icône de la page
st.set_page_config(page_title="Swift loader", layout="wide", page_icon=im, initial_sidebar_state="collapsed")  # Définir la configuration de la page

# Titre et message de bienvenue pour la page streamlit
st.title('AI Assistant for Data Science 🤖')  # Titre de la page
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")  # Message de bienvenue

# Barre latérale d'explication
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a File.*')  # Message explicatif
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')  # Message explicatif détaillé
    st.divider()  # Ajouter un séparateur
    st.caption("<p style ='text-align:center'> made with ❤️ by leslie</p>", unsafe_allow_html=True)  # Message de crédit

# Initialiser les états de session
if "past" not in st.session_state:
    st.session_state["past"] = []  # Historique des questions de l'utilisateur
if "generated" not in st.session_state:
    st.session_state["generated"] = []  # Réponses générées par l'assistant
if "source" not in st.session_state:
    st.session_state["source"] = []  # Sources des réponses
if "input" not in st.session_state:
    st.session_state["input"] = ""  # Entrée de l'utilisateur
if "temp" not in st.session_state:
    st.session_state["temp"] = ""  # Entrée temporaire
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []  # Sessions stockées
if "lang" not in st.session_state:
    st.session_state["lang"] = "English"  # Langue par défaut
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.05  # Température par défaut pour la génération de texte
if "decoding" not in st.session_state:
    st.session_state["decoding"] = "greedy"  # Méthode de décodage par défaut
if "LLM" not in st.session_state:
    st.session_state["LLM"] = "ibm/granite-13b-chat-v2"  # Modèle de langage par défaut
if "max_token" not in st.session_state:
    st.session_state["max_token"] = 4096  # Nombre maximal de tokens
if "min_token" not in st.session_state:
    st.session_state["min_token"] = 5  # Nombre minimal de tokens
if "sim_vector"not in st.session_state:
    st.session_state["sim_vector"] = 3  # Nombre de vecteurs de similarité

langs = ["English", "French"]  # Langues disponibles
Decoding_met = ['greedy', 'sample']  # Méthodes de décodage disponibles

# Définir les invites spécifiques à chaque langue
language_prompts = {
    "French": {
        "TITLE": "Recherche documentaire augmentée avec watsonx.ai",  # Titre en français
        "PROMPTSTUB": "Salut 😄, Allez y poser votre question ici...?",  # Texte d'invite en français
        "LEFTABOUT": "Recherche documentaire dans un corpus de documents spécifiques, augmentée avec watsonx.ai"  # Description en français
    },
    "English": {
        "TITLE": "Men Spec Assistant",  # Titre en anglais
        "PROMPTSTUB": "Hello 😄, Go ahead and ask your question here...?",  # Texte d'invite en anglais
        "LEFTABOUT": "Illustrate how we can easily and quickly identify the compatibility impacts of product upgrade with watsonx.ai"  # Description en anglais
    }
}

selected_lang = st.session_state["lang"]  # Langue sélectionnée
TITLE = language_prompts[selected_lang]["TITLE"]  # Titre correspondant à la langue sélectionnée
PROMPTSTUB = language_prompts[selected_lang]["PROMPTSTUB"]  # Texte d'invite correspondant à la langue sélectionnée
LEFTABOUT = language_prompts[selected_lang]["LEFTABOUT"]  # Description correspondant à la langue sélectionnée

# Définir les éléments de l'interface utilisateur
def clear_text():
    st.session_state["temp"] = st.session_state["input"]  # Sauvegarder l'entrée de l'utilisateur dans une variable temporaire
    st.session_state["input"] = ""  # Réinitialiser l'entrée de l'utilisateur

def get_text():
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder=PROMPTSTUB,
        on_change=clear_text,
        label_visibility="hidden"
    )  # Champ de saisie pour l'utilisateur avec un texte d'invite et une fonction de réinitialisation
    input_text = st.session_state["temp"]  # Récupérer l'entrée temporaire
    st.session_state["temp"] = ""  # Réinitialiser l'entrée temporaire
    return input_text  # Retourner l'entrée de l'utilisateur

def new_chat():
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
        save.append(st.session_state["source"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["source"] = []
    st.session_state["input"] = ""
    st.session_state["temp"] = ""

def clear_history():
    st.session_state["stored_session"] = []

def restart_chat():
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["source"] = []
    st.session_state["input"] = ""
    st.session_state["temp"] = ""
    # Réinitialiser d'autres paramètres si nécessaire
    # Par exemple : st.session_state["temperature"] = 0.05
def clear_history():
    if st.session_state.stored_session:
        st.session_state.stored_session = []  # Effacer l'historique des sessions stockées

# Rules data
rule1 = """
A drone can be operated in the "Open "category if:
- The drone has one of the class identification labels 0, 1, 2, 3, or 4.
- The drone was purchased before 1 January 2023, with no class identification label as above.
- The drone has a maximum take-off mass of less than 25 kg (55 lbs).
- The remote pilot keeps the drone at a safe distance away from people.
- The drone will not be operated directly over people unless it has a class identification label or is lighter than 250 g (0.55 lbs). (Please refer to subcategories of operations: A1, A2, and A3 to find out where you can fly with your drone).
- The remote pilot will maintain a visual line of sight (VLOS) or the remote pilot will be assisted by a UA observer.
- The remote pilot will not operate the drone above 120m (400ft).
- The drone will not carry any dangerous goods and will not drop any material.
"""

rule2 = """
General Rules for Flying a Drone in France
- All drones of 800g or more must be registered by their owner on AlphaTango, the public portal for users of remotely piloted aircraft. The drone then receives a registration number that must be affixed permanently, visibly, on the drone and must allow reading at a distance of 30 centimeters, with the naked eye.  The drone pilot must be able to provide proof of registration in the event of a check.
- Drone pilots must always maintain a line of sight with their drones. If a visual observer is tracking the drone, the pilot may fly out of his or her own range of sight.
- Drones may not be flown at night (unless with special authorization from the local prefect).
- Drones may not be flown over people; over airports or airfields; over private property (unless with owner's authorization); over military installations, prisons, nuclear power plants, historical monuments, or national parks. Use this map to locate flight restrictions by geolocation.
- Drones may also not be flown over ongoing fires, accident zones, or around emergency services.
- Drones may not be flown above 150 meters (492 feet), or higher than 50 meters (164 feet) above any object or building that is 100 meters (328 feet) or more in height.
"""

rule3 = """
Rules for Flying a Drone Commercially in France
- Drone pilots who fly for purposes other than leisure (commercial drone pilots) must pass a theoretical exam. The exam can be taken online or at specified DSAC facilities. Procedures for taking this exam are described on this page. Upon passing the exam, the pilot will receive a theoretical telepilote certificate. The pilot must have this printed and with them during all flights.
- Commercial drone pilots must also undergo basic practical training. The operator must define and provide the necessary additional training, considering the types of aircraft they use and the specific activities they perform. At the end of the training, the training organizations will provide the telepilots with a training follow-up certificate for the corresponding scenarios.
- A drone pilot cannot provide his own practical training.
"""

rule4 = """
Rules for Flying a Drone for Recreation in France
Based on our research, here are the additional requirements to fly a drone for recreation in France.
- Drone pilots who fly for leisure or recreation only do not need a training certificate when their drone's mass is less than 800 grams.
- Drone pilots operating a remotely piloted aircraft of 800g or more for recreational purposes must undergo training. This training can be: (1) the Fox AlphaTango training offered by the DGAC or (2) training provided by the FFAM or UFOLEP recognized as equivalent by the DGAC.
"""

rules = [rule1, rule2, rule3, rule4]

legal_classes = """
Class 0 legal requirements
- Have a MTOM (Maximum Takeoff Mass) less than or equal to 250 g.
- Have a maximum horizontal flight speed of 19 m/s.
- Have limited altitude from the takeoff point to 120 m.
- Be powered by electricity.
Class 1 legal requirements
- Have a MTOM (Maximum Takeoff Mass) less than or equal to 900 g or that the energy transmitted in case of impact is less than or equal to 80 J.
- Have a maximum horizontal flight velocity of 19 m/s.
- Have limited altitude from the point of takeoff to 120 m.
- Be powered by electricity.
- Have a unique serial number.
- Have a system of direct distance identification.
- Be equipped with a geolocation system.
- Be equipped with a low battery warning system for the UA (Unmanned Aircraft) and the control station (CS).
- Equip lights for activity control and nocturnal flight (intermittent green light).
Class 2 legal requirements
- Have a MTOM less than or equal to 4 kg.
- Have limited altitude from the takeoff point to 120 m.
- Be powered by electricity.
- Be equipped with a data link protected against unauthorized access to the control functions (C2).
- Except if it is a UA of the toy type, be equipped with a selectable low-speed mode that limits the speed to 3 m/s maximum.
- Have a serial number.
- Have a direct distance identification system.
- Be equipped with a geolocation system.
- Have a low battery warning system for the UA and the control station (CS).
- Equip lights for activity control and intermittent green flight (flashing green light).
Class 3 legal requirements
- Have a MTOM (Maximum Takeoff Mass) of less than or equal to 25 kg and a maximum characteristic dimension of less than or equal to 3 meters.
- Have a maximum altitude limited to 120 meters from the takeoff point.
- Be powered by electricity.
- Have a unique serial number.
- Have a system for direct distance identification.
- Be equipped with a geolocation system.
- Be equipped with a low battery warning system for the UA (Unmanned Aircraft) and the control station (CS).
- Be equipped with lights for attitude control and nocturnal flight (intermittent green light).
Class 4 legal Requirements
- Have a MTOM (Maximum Takeoff Mass) less than 25 kg, including the payload.
- Not have automatic control modes, except for assistance in flight stabilization without any direct effect on the trajectory and for assistance in case of loss of link, provided that there is a predetermined fixed position of the flight controls in case of loss of the link.
- Be intended for the practice of aeromodelling.
Class 5 legal Requirements
- Have a MTOM less than or equal to 25 kg.
- Not be a fixed wing UA, if it is a captive UA.
- Provide the pilot with clear and concise altitude information of the UA.
- Be equipped with a selectable low-speed mode that limits the speed to 5 m/s maximum.
- Before a data link loss (C2), count on a method to recover or to finish the flight safely.
- Have a method of link recovery from the control station (C2), in case of failure, a termination system for safe flight.
- Be equipped with a data link protected against unauthorized access to control functions (C2).
- Be powered by electricity.
- Have a unique serial number.
- Have a system for direct distance identification.
- Be equipped with a geolocation system.
- Be equipped with a low battery warning system for the UA and the control station (CS).
- Equip lights for activity control and nocturnal flight.
- If the UA has an access limitation function to certain zones or air volumes, this should be inoperable from the control system, and the pilot must be informed when it is impeded to enter the UA into certain zones or air volumes.
- A class C5 UA that assists in a class C3 UA that has installed a distance pilot accessory should convert the class C3 UA into a class C5 UA. The accessory kit should not include changes in the UA from class C3.
Class 6 legal Requirements
- Have a MTOM less than or equal to 25 kg.
- Have a system that provides the pilot with clear and concise information about the UA's altitude, ensuring that the UA does not exceed the horizontal and vertical limits of a programmable operational volume.
- Have a velocity reduction mechanism that limits groundspeed to a maximum of 50 m/s.
- Before a data link loss (C2), count on a method to recover or terminate the flight safely.
- Have a method of link recovery from the control station (C2), or in case of failure, a termination system for safe flight.
- Be powered by electricity.
- Have a unique serial number.
- Have a system for direct distance identification.
- Be equipped with a geolocation system.
- Be equipped with a low battery warning system for the UA and the control station (CS).
- If the UA has a function to limit access to certain areas or volumes of airspace, it should be inoperable from the control system, and the pilot must be informed at a distance when it is prevented from entering the UA into those areas or airspace volumes.
- Equip lights for attitude control and nocturnal flight."""

legal_chunks = ["Class " + item for item in legal_classes.split("Class")]
legal_chunks.pop(0)
        
prompt_template_1 = (
            """[INST]
                
            context :  {context}
            
           
            ---
            <<SYS>>you are a banking expert.You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
            f a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            extract entities from the context. do not generate extra informations.
            the context is a document talking about drones.
            <</SYS>>  
           
            Input: regarding the context  ,{question}
            Output:[/INST]"""
        )
        
prompt_template_2 = (
    """[INST]
    You are a helpful assistant helping people working on industrial products. An industrial product is a system of systems (SoS). Each system is described by a list of requirements.

    Requirements can be of the following types:
    - Legal requirements: Legal requirements are rules and norms from the authorities. The system must comply with all the legal requirements
    - Stakeholder requirements: Stakeholder requirements express the intended interaction the system will have with its operational environment, including stakeholders
    - System requirements: System requirements are all of the requirements at the system level that describe the functions which the system as a whole should fulfill to satisfy the stakeholder requirements
    - Subsystem requirements: Subsystem requirements are all of the requirements of a subsystem. They are linked to at least one system requirement.

    There is a bidirectional relationship between requirement types.
    - A stakeholder requirement is satisfied by one or multiple system requirements
    - A system requirement is satisfied by one or multiple subsystem requirements

    Given a list of requirements that are linked together, answer the user.

    List of requirements:

    {context}

    User: For stakeholder and system requirements only, to what extent is each requirement satisfied?
    - Not satisfied
    - Partially satisfied
    - Totally satisfied

    Answer using a table with the following format:
    | Requirement Type | Requirement | Satisfaction |

    Provide an explanation and a list of possible corrections.
    [/INST]"""
)
        
prompt_template_3 = (
            """[INST]
            context :  {context}
            
           
            ---
            <<SYS>>you are a expert in legal document.You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
            f a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            extract entities from the context. do not generate extra informations.
            the context is a french legal document on drones.answer must be only in french 
            <</SYS>>  
           
            Input: regarding the context  ,{question}
            Output:[/INST]"""
        )
prompt_template_4 = (
            """### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.
            Answer the question based on the context.the context is uncluded between --context start-- and  --context end--
            Complete the conversation below between [|Human|] and [|AI|]:
            ### Input: [|Human|] --context start--  {context}  --context end--
            ### Input: [|Human|] {question}
            ### Response: [|AI|] """
        )

prompt_template = (
                """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 


            Context :{context}

        Question: {question}
        Answer in """ + st.session_state["lang"] + ":"
        )

prompt_template = (
            """[INST]
            context :  {context}
           
            ---
            <<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
            f a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            extract entities from the context. do not generate extra informations.anwser must be in english.
            Use the provided rules and legal requirements when answering questions about drone operations and regulations.
            <</SYS>>  
           
            Input: Regarding the context, rules, and legal requirements, {question}
            Output:[/INST]"""
        )
prompt_template_5 = (
            """
            Given the document and the current conversation between a user and an agent, your task is as follows: Answer any user query by using information from the document. The response should be detailed.
            If you don't know the answer to a question, please don't share false information.
            extract entities from the context. do not generate extra informations.anwser must be in english.
            DOCUMENT: {context}
            DIALOG: USER: {question}
            """

        )
prompt_template_llama = ("""
    You are an AI assistant analyzing a mission report about drone operations. Use only the information provided in the context to answer the questions the user will type. If the information is not in the context, say you don't have that information. Do not make up or infer any details not explicitly stated in the report.Your task is to provide ONLY answers to questions, never ask questions yourself.

    IMPORTANT INSTRUCTIONS:
    1. Use only the information provided in the context to answer the user's question.
    2. If the information is not in the context, say "I don't have that information in the provided context."
    3. DO NOT make up or infer any details not explicitly stated in the report.
    4. DO NOT ask any questions under any circumstances.
    5. Generate ONLY the answer to the user's question.

    PENALTY: If you generate a question instead of an answer, your response will be considered incorrect and discarded.

    Context: {context}

    User Question: {question}
    Your Answer (remember, ONLY provide the answer, DO NOT ask questions):
    """
)

prompt_template_granite = (

    """
    You are an AI assistant analyzing requirements for an industrial product, which is a system of systems (SoS). Use only the information provided in the context to answer the user's questions. If the information is not in the context, state that you don't have that information. Do not make up or infer any details not explicitly stated.

    IMPORTANT INSTRUCTIONS:
    1. Use only the information provided in the context to answer the user's question.
    2. If the information is not in the context, say "I don't have that information in the provided context."
    3. DO NOT make up or infer any details not explicitly stated.
    4. DO NOT ask any questions under any circumstances.
    5. Generate ONLY the answer to the user's question.
    6. Provide a comprehensive analysis of the requirements, including stakeholder requirements, system requirements, and legal requirements.
    7. Use proper grammar and punctuation.
    8. Generate only safe and respectful content.
    9. Ensure all content is factually accurate and relevant to the prompt.
    10. Format the answer as follows:
        - List all stakeholder and system requirements
        - Provide any relevant legal requirements or rules
        - Analyze whether the system complies with the legal requirements
        - Explain why the system does or does not comply
        - If it does not comply, suggest possible corrections

    PENALTY: If you generate a question instead of an answer, your response will be considered incorrect and discarded.

    Context:
    {context}

    User Question: {question}
    Your Answer (remember, ONLY provide the answer, DO NOT ask questions):
    """
)

prompt_template_vol =(
    """
    You are an AI assistant helping analyze requirements for an industrial product, which is a system of systems (SoS). Each system is described by a list of requirements.

    Requirements can be of the following types:
    - Stakeholder Requirements: Describe what users need from the system
    - Legal Requirements: Regulatory rules and norms the system must comply with  
    - System Requirements: Functionality needed to satisfy stakeholder requirements

    There is a bidirectional relationship between requirement types:
    - A Stakeholder Requirement is satisfied by one or multiple System Requirements
    - A System Requirement satisfies at least one Stakeholder Requirement
    - A System Requirement can be satisfied by one or multiple System Requirements 1
    - All requirements must comply with Legal Requirements

    Requirements should be clear, concise (<300 characters), and avoid hidden dependencies.

    Given a list of requirements from the CSV data, analyze them for quality and compliance. 

    Specifically:
    1. Identify the type of each requirement (Stakeholder, Legal, System, System 1)
    2. Check if system requirements properly satisfy stakeholder requirements
    3. Verify compliance with any legal requirements
    4. Flag any requirements that are unclear, too long, or have hidden dependencies
    5. Suggest improvements to problematic requirements

    Provide a concise summary of your analysis, highlighting key issues and recommendations.

    CSV data:
    {context}

    Question: {question}

    Answer (remember, ONLY provide the answer, DO NOT ask questions):
    
    """
)


# New chat and clear history buttons
with st.sidebar.expander("🛠️ Tools", expanded=False):
    st.button("New Chat", on_click=new_chat, type="primary", key="new_chat_expander")
    st.button("Clear History", on_click=clear_history, type="primary", key="clear_history_expander")     

        
# Sélection de la langue dans la barre latérale
with st.sidebar.expander("Language", expanded=False):
    selected_lang = st.selectbox("Choose your language", options=langs, key="language_selector")
    st.session_state["lang"] = selected_lang  # Met à jour la langue sélectionnée dans l'état de la session
    

# Éléments de la barre latérale
with st.sidebar:
    st.markdown("---")  # Ajoute une ligne de séparation
    st.markdown("# About")  # Titre "About"
    st.markdown(TITLE)  # Affiche le titre basé sur la langue sélectionnée
    st.markdown("Use Case : ")  # Texte "Use Case"
    st.markdown(LEFTABOUT)  # Affiche la description basée sur la langue sélectionnée
    st.markdown("---")  # Ajoute une ligne de séparation
    st.button("Restart chat", on_click=new_chat)  # Bouton pour redémarrer la conversation
    file_path = st.file_uploader("Document to upload")  # Chargement de fichier

    st.selectbox("Select a model", options=llms, key='LLM')  # Sélecteur de modèle
    st.selectbox("Select your decoding method", options=Decoding_met, key="decoding")  # Sélecteur de méthode de décodage
    
    if st.session_state["decoding"] == "sample":
        st.slider("Temperature", min_value=0.05, max_value=2.0, step=0.01, key="temperature")  # Curseur pour régler la température de décodage
    
    st.number_input("Min Tokens", min_value=0, max_value=500, key='min_token')  # Entrée pour le nombre minimum de tokens
    st.number_input("Max Tokens", min_value=1, max_value=1536, key='max_token')  # Entrée pour le nombre maximum de tokens
    st.number_input("Similarity vectors", min_value=1, max_value=8000, key='sim_vector')  # Entrée pour le nombre de vecteurs de similarité

    # Dictionnaire des modèles de prompt
    prompt_templates = {
        "swift": prompt_template_1,
        "discrepancies": prompt_template_2,
        "granite": prompt_template_3,
        "jais": prompt_template_4,
        "default": prompt_template,
        "legal": prompt_template_5,
        "drone": prompt_template_granite,
        "llama": prompt_template_llama,
        "vol": prompt_template_vol
    }

    template_options = list(prompt_templates.keys())  # Liste des clés des modèles de prompt
    template_choice = st.selectbox("Select a prompt template", options=template_options)  # Sélecteur de modèle de prompt
    selected_template = prompt_templates[template_choice]  # Modèle de prompt sélectionné

# Fonction pour obtenir la clé API GenAI, mise en cache
@st.cache_data
def get_gen_api_key():
    load_dotenv(find_dotenv())  # Charger les variables d'environnement depuis le fichier .env
    assert os.environ.get("GENAI_KEY") is not None, "Missing GENAI_KEY"  # Vérifier que la clé API existe
    return os.environ.get("GENAI_KEY")  # Retourner la clé API

# Fonction pour obtenir l'endpoint API GenAI, mise en cache
@st.cache_data
def get_gen_api_endpoint():
    load_dotenv(find_dotenv())  # Charger les variables d'environnement depuis le fichier .env
    assert os.environ.get("GENAI_ENDPOINT") is not None, "Missing GENAI_ENDPOINT"  # Vérifier que l'endpoint API existe
    return os.environ.get("GENAI_ENDPOINT")  # Retourner l'endpoint API

GENAI_KEY = "pak-8qThaPGnrxj9W9AQTaMLDM5VHwRy5NHBxkMOprQQcVk" 
GENAI_ENDPOINT = "https://bam-api.res.ibm.com/"  
# Fonction pour initialiser la chaîne de QA, mise en cache
@st.cache_resource
def initqachain(lang, decoding, temperature, llm_list, min_token, max_token, k, selected_template):
    if GENAI_KEY and GENAI_ENDPOINT:
        model = LangChainInterface(
            model_id=llm_list,
            client=Client(credentials=Credentials(GENAI_KEY, api_endpoint=GENAI_ENDPOINT)),
            parameters=TextGenerationParameters(
                decoding_method=DecodingMethod.GREEDY if decoding == "greedy" else DecodingMethod.SAMPLE,
                temperature=temperature,
                max_new_tokens=max_token,
                min_new_tokens=min_token,
                repetition_penalty=1.1,  # Légèrement réduit pour permettre plus de variation
                stop_sequences=["\n\n"]
            )
        )

        customprompt = PromptTemplate(
            template=selected_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": customprompt}
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",  # Utiliser la méthode "stuff" pour les documents plus courts
            retriever=ensemble_retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        
        return qa
    else:
        st.sidebar.warning("GENAI API key and endpoint required to use this application.")  # Avertissement si les clés API manquent

print("file ")
print(file_path)
print("------")


def create_pandas_agent(df):
    # Crée un modèle d'interface LangChain avec les paramètres définis dans l'état de la session
    model = LangChainInterface(
        model_id=st.session_state["LLM"],
        client=Client(credentials=Credentials(GENAI_KEY, api_endpoint=GENAI_ENDPOINT)),
        parameters=TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY,  # Méthode de décodage définie sur GREEDY
            temperature=st.session_state["temperature"],  # Température pour la génération de texte
            max_new_tokens=st.session_state["max_token"],  # Nombre maximal de nouveaux tokens
            min_new_tokens=st.session_state["min_token"],  # Nombre minimal de nouveaux tokens
            repetition_penalty=2,  # Pénalité pour la répétition
            stop_sequences=["\n"]  # Séquence d'arrêt
        )
    )
    # Définir un prompt personnalisé pour les opérations sur le dataframe pandas
    custom_prompt = """You are working with a pandas dataframe in Python. The dataframe is called 'df'.
    Perform operations as requested. Always return numerical results when applicable.
    If you're unsure about a column name, check df.columns first."""
    
    # Crée et retourne un agent pour le dataframe pandas
    return create_pandas_dataframe_agent(
        model, 
        df, 
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},  # Gérer les erreurs de parsing
        prefix=custom_prompt
    )

def parse_operation(query, df):
    query = query.lower()  # Convertir la requête en minuscules
    
    # Vérifier les opérations spécifiques dans la requête
    if 'sum' in query or 'total' in query:
        column = extract_column(query, df)  # Extraire le nom de la colonne
        if column:
            return f"df['{column}'].sum()"  # Retourner l'opération de somme sur la colonne
    elif 'mean' in query or 'average' in query:
        column = extract_column(query, df)  # Extraire le nom de la colonne
        if column:
            return f"df['{column}'].mean()"  # Retourner l'opération de moyenne sur la colonne
    elif 'max' in query:
        column = extract_column(query, df)  # Extraire le nom de la colonne
        if column:
            return f"df['{column}'].max()"  # Retourner l'opération de maximum sur la colonne
    elif 'min' in query:
        column = extract_column(query, df)  # Extraire le nom de la colonne
        if column:
            return f"df['{column}'].min()"  # Retourner l'opération de minimum sur la colonne
    elif 'count' in query:
        return "len(df)"  # Retourner le nombre de lignes du dataframe
    elif 'describe' in query:
        return "df.describe()"  # Retourner les statistiques descriptives du dataframe
    elif any(op in query for op in ['+', '-', '*', '/']):
        return parse_arithmetic(query, df)  # Analyser les opérations arithmétiques dans la requête
    
    return None  # Retourner None si aucune opération n'est trouvée

def extract_column(query, df):
    for col in df.columns:
        if col.lower() in query:
            return col  # Retourner le nom de la colonne si elle est trouvée dans la requête
    return None  # Retourner None si aucune colonne n'est trouvée

def parse_arithmetic(query, df):
    operators = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}  # Dictionnaire des opérateurs arithmétiques
    for op, func in operators.items():
        if op in query:
            cols = [col for col in df.columns if col.lower() in query.lower()]  # Trouver les colonnes dans la requête
            if len(cols) >= 2:
                return f"df['{cols[0]}'].{func}(df['{cols[1]}'])"  # Retourner l'opération arithmétique sur les colonnes
    return None  # Retourner None si l'opération arithmétique n'est pas trouvée
    
def displayPDF(file_path, page_num):
    with open(file_path, 'rb') as file:  # Open the file in binary read mode
        file_bytes = io.BytesIO(file.read())  # Read the file into a bytes buffer
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")  # Open the document as a PDF
        page = pdf_document.load_page(page_num - 1)  # Load the specified page
        pix = page.get_pixmap()  # Get the pixmap (image) of the page
        image_base64 = base64.b64encode(pix.tobytes()).decode("utf-8")  # Encode the image in base64
        image_html = f'<img src="data:image/png;base64,{image_base64}" width="700" height="1000"/>'  # Create the HTML for the image
        return image_html  # Return the HTML string

def read_csv_file(file_path):
    try:
        # Essayer de lire le CSV avec le délimiteur par défaut (virgule)
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
          
        df = pd.read_csv(file_path, delimiter=';')  # Essayer avec le délimiteur point-virgule
    return df  # Retourner le dataframe

def process_csv(df):
    st.write("## CSV Data Processing")  # Titre de la section de traitement des données CSV

    # Informations de base
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")  # Afficher le nombre de lignes
    st.write(f"Number of columns: {df.shape[1]}")  # Afficher le nombre de colonnes

    # Afficher les noms de colonnes et les types de données
    st.write("### Column Names and Data Types")
    st.write(df.dtypes)  # Afficher les types de données des colonnes

    # Statistiques descriptives
    st.write("### Summary Statistics")
    st.write(df.describe())  # Afficher les statistiques descriptives du dataframe

    # Valeurs manquantes
    st.write("### Missing Values")
    missing_values = df.isnull().sum()  # Calculer le nombre de valeurs manquantes par colonne
    st.write(missing_values)  # Afficher les valeurs manquantes

    # Afficher les premières lignes
    st.write("### First Few Rows")
    st.dataframe(df.head())  # Afficher les premières lignes du dataframe

    # Carte de chaleur de la corrélation
    st.write("### Correlation Heatmap")
    numeric_columns = df.select_dtypes(include=[np.number]).columns  # Sélectionner les colonnes numériques
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))  # Créer une figure pour la carte de chaleur
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)  # Créer la carte de chaleur des corrélations
        st.pyplot(fig)  # Afficher la figure

    # Sélection de colonnes pour analyse approfondie
    st.write("### Column Analysis")
    selected_columns = st.multiselect("Select columns for further analysis:", df.columns)  # Sélectionner les colonnes à analyser
    if selected_columns:
        st.write("Selected Columns:")
        st.dataframe(df[selected_columns])  # Afficher les colonnes sélectionnées

        # Graphiques de distribution pour les colonnes numériques
        numeric_selected = df[selected_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_selected) > 0:
            st.write("### Distribution of Numeric Columns")
            for col in numeric_selected:
                fig, ax = plt.subplots()  # Créer une figure pour le graphique de distribution
                sns.histplot(df[col], kde=True, ax=ax)  # Créer le graphique de distribution avec une courbe KDE
                ax.set_title(f'Distribution of {col}')  # Définir le titre du graphique
                st.pyplot(fig)  # Afficher la figure

    # Options de nettoyage des données
    st.write("### Data Cleaning Options")
    if st.checkbox("Remove rows with missing values"):
        df = df.dropna()  # Supprimer les lignes avec des valeurs manquantes
        st.write("Rows with missing values removed.")

    if st.checkbox("Fill missing values with mean"):
        df = df.fillna(df.mean())  # Remplir les valeurs manquantes avec la moyenne
        st.write("Missing values filled with mean.")

    # Options de transformation des données
    st.write("### Data Transformation Options")
    for column in df.select_dtypes(include=[np.number]).columns:
        if st.checkbox(f"Normalize {column}"):
            df[f"{column}_normalized"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())  # Normaliser la colonne
            st.write(f"{column} normalized.")

    # Afficher le dataframe mis à jour
    st.write("### Updated Dataframe")
    st.dataframe(df)  # Afficher le dataframe mis à jour

    return df  # Retourner le dataframe mis à jour

if file_path:
    file_path.seek(0)  # Reset the file pointer to the beginning
    vector_store, ensemble_retriever, temp_file_path = vector_db_creation(file_path)  # Create vector store and retrievers
    file_extension = os.path.splitext(file_path.name)[1].lower()  # Get the file extension in lowercase
    
    if file_extension == '.pdf':
        num_pages = fitz.open(temp_file_path).page_count  # Get the number of pages in the PDF
        st.write(f"PDF file with {num_pages} pages.")  # Display the number of pages
        st.markdown(displayPDF(temp_file_path, 1), unsafe_allow_html=True)  # Display the first page of the PDF
    elif file_extension == '.csv':
        df = read_csv_file(temp_file_path)  # Read the CSV file
        df = process_csv(df)  # Process the CSV file
        pandas_agent = create_pandas_agent(df)  # Create a Pandas agent
        st.dataframe(df)  # Display the DataFrame
        st.write("Dataframe Info:")  # Display DataFrame info
        st.write(f"Shape: {df.shape}")  # Display DataFrame shape
        st.write("Columns:")  # Display DataFrame columns
        st.write(df.columns.tolist())  # List DataFrame columns
        st.write("Data Types:")  # Display DataFrame data types
        st.write(df.dtypes)  # Display DataFrame data types
        st.write("Summary Statistics:")  # Display DataFrame summary statistics
        st.write(df.describe())  # Display DataFrame summary statistics
        # Add a text input for user queries about the DataFrame
        user_query = st.text_input("Ask a question about the dataframe:")
        if user_query:
            try:
                result = pandas_agent.run(user_query)  # Run the user query with the Pandas agent
                st.write("Answer:", result)  # Display the answer
            except Exception as e:
                st.write(f"The agent encountered an error: {str(e)}")  # Display an error message if the agent fails
                st.write("Falling back to basic dataframe operations...")  # Fall back to basic DataFrame operations
                try:
                    operation = parse_operation(user_query, df)  # Parse the operation requested by the user
                    if operation:
                        result = eval(operation)  # Evaluate and execute the operation
                        st.write(f"Result of operation '{operation}':")  # Display the result of the operation
                        if isinstance(result, (int, float)):
                            st.write(f"{result:,}")  # Add thousand separators if the result is a number
                        else:
                            st.write(result)  # Display the result
                    else:
                        st.write("I couldn't interpret that query. Please try rephrasing or specifying a clear operation.")  # Message if the operation is not interpreted
                except Exception as e2:
                    st.write(f"Fallback also failed: {str(e2)}")  # Display an error message if the fallback fails
                    st.write("Please try rephrasing your question or asking about specific columns.")  # Ask to rephrase the question
    elif file_extension == '.txt':
        content = file_path.read().decode('utf-8')  # Read the content of the text file
        st.text(content)  # Display the content of the text file
    # Remove the temporary file when done
    os.remove(temp_file_path)   
    
st.title(TITLE)  # Afficher le titre de la page

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """  # CSS pour masquer le menu principal et le pied de page
st.markdown(hide_default_format, unsafe_allow_html=True)  # Appliquer le CSS

user_input = get_text()  # Obtenir l'entrée de l'utilisateur

if user_input:
    context = "Provide the list of requirements or document context here"  # Contexte à utiliser pour répondre aux questions
    question = user_input  # Question posée par l'utilisateur
    qa = initqachain(st.session_state["lang"], st.session_state["decoding"], st.session_state["temperature"], st.session_state["LLM"], st.session_state['min_token'], st.session_state['max_token'], st.session_state["sim_vector"], selected_template)  # Initialiser la chaîne de questions-réponses
    #result = qa({"context": context, "question": question})
    result = qa({"query": question})  # Obtenir la réponse à la question
    output = result["result"]  # Extraire le résultat
    docs = result["source_documents"]  # Extraire les documents source
    source = (
        "\n ".join(
            list(
                map(
                    lambda d: f"{os.path.basename(d.metadata['source'])}, page {d.metadata.get('page', 'N/A')}",
                    docs,
                )
            )
        ) 
                + "."
    )  # Générer une chaîne de caractères listant les sources
    st.session_state["past"].append(user_input)  # Ajouter la question à l'historique
    st.session_state["generated"].append(output)  # Ajouter la réponse générée à l'historique
    st.session_state["source"].append(source)  # Ajouter les sources à l'historique

# Permettre le téléchargement de la conversation
download_data = []

def generate_blog_post(topic, vector_store, k):
    docs = vector_store.similarity_search(topic, k=k)  # Rechercher des documents similaires dans la base de données vectorielle
    output = [{'content': doc.page_content, 'page': doc.metadata.get('page', 'N/A'), 'doc_path': doc.metadata['source'], 'doc_source': "document: " + str(os.path.basename(doc.metadata['source'])) + " \npage: " + str(doc.metadata.get('page', 'N/A'))} for doc in docs]
    return output  # Retourner une liste de documents similaires avec leurs métadonnées

def display_source_image(doc_path, page_num, content=None):
    if doc_path.endswith(".pdf"):
        return displayPDF(doc_path, page_num)  # Afficher une page du document PDF
    elif doc_path.endswith(".csv"):
        return display_csv_source(doc_path, content)  # Utiliser la nouvelle fonction pour les CSV
    elif doc_path.endswith(".txt"):
        with open(doc_path, 'r') as f:
            return f.read()  # Lire et retourner le contenu du fichier texte

def display_csv_source(file_path, content):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # Find rows that contain the content
    relevant_rows = df[df.apply(lambda row: content in ' '.join(row.astype(str)), axis=1)]
    
    if not relevant_rows.empty:
        st.write("Relevant data from CSV:")
        st.dataframe(relevant_rows)
    else:
        st.write("Couldn't find exact match. Showing first few rows:")
        st.dataframe(df.head())

    # Show column names
    st.write("Columns used:")
    st.write(df.columns.tolist())
    
    return df  # Return the dataframe for consistency with the original function

with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.text_area("🧐 question ", st.session_state["past"][i], key="Question_" + str(i), disabled=True)  # Afficher la question de l'utilisateur
        st.success(st.session_state["generated"][i], icon="🤖")  # Afficher la réponse générée
        st.info(st.session_state["source"][i], icon="📄")  # Afficher les sources
        for count, r in enumerate(generate_blog_post(st.session_state["past"][i], vector_store, st.session_state["sim_vector"])):
            st.write(f"Displaying source: {r['doc_path']}, Page: {r['page']}")  # Afficher le chemin du document source et la page
            if r["doc_path"].endswith(".pdf"):
                st.markdown(displayPDF(temp_file_path, int(r['page']) + 1), unsafe_allow_html=True)  # Afficher une page du document PDF
            elif r["doc_path"].endswith(".csv"):
                display_csv_source(r["doc_path"], r['content'])  # Utiliser la nouvelle fonction pour les CSV
            elif r["doc_path"].endswith(".txt"):
                with open(r["doc_path"], 'r') as f:
                    st.text(f.read())  # Lire et afficher le contenu du fichier texte
        download_data.append(f"User: {st.session_state.past[i]}")  # Ajouter la question de l'utilisateur aux données à télécharger
        download_data.append(f"Bot: {st.session_state.generated[i]}")  # Ajouter la réponse de l'assistant aux données à télécharger
        download_data.append(f"source: {st.session_state.source[i]}")  # Ajouter les sources aux données à télécharger
        
    download_str = "\n".join(download_data)  # Convertir les données de la conversation en chaîne de caractères
    if download_str:
        st.download_button("Download conversation", download_str)  # Ajouter un bouton pour télécharger la conversation

for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)  # Afficher les sessions de conversation stockées dans la barre latérale
