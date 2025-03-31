# Assistant IA pour la Science des Données

## Aperçu

Ce projet est une application web interactive conçue pour aider les utilisateurs dans leurs projets de science des données. Construite avec Streamlit, elle exploite divers modèles d'IA pour aider les utilisateurs à analyser des données, répondre à des questions et fournir des insights. L'application prend en charge plusieurs types de fichiers, notamment PDF, CSV et fichiers texte, et utilise des modèles de langage de pointe pour générer des réponses basées sur les requêtes des utilisateurs.

## Fonctionnalités

- **Interface utilisateur interactive** : Construite avec Streamlit pour une interface conviviale.
- **Téléchargement de fichiers** : Prend en charge le téléchargement et le traitement de fichiers PDF, CSV et texte.
- **Modèles de langage** : Utilise divers modèles de langage pour la génération de texte et la réponse aux questions.
- **Visualisation de données** : Intègre Seaborn et Matplotlib pour la visualisation de données.
- **Gestion des sessions** : Suit les interactions des utilisateurs et maintient l'historique des sessions.

## Prérequis

- Python 3.10+
- Streamlit
- LangChain
- GenAI
- HuggingFace
- PyMuPDF (fitz)
- Pandas
- Numpy
- Seaborn
- Matplotlib
- dotenv

## Installation

1. Clonez le dépôt :

   ```sh
   git clone https://github.com/votre-repo/pdf-rag-main.git
   cd pdf-rag-main
   ```

2. Créez un environnement virtuel :

   ```sh
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. Installez les paquets requis :

   ```sh
   pip install -r requirements.txt
   pip install -r requirements-app.txt
   ```

4. Configurez les variables d'environnement :
   Il y a un fichier `.env` dans le répertoire `app`. Assurez-vous qu'il contient les variables d'environnement nécessaires :
   ```env
   GENAI_KEY=votre_cle_api_genai
   GENAI_ENDPOINT=https://votre_endpoint_genai
   ```

## Utilisation

1. Lancez l'application Streamlit :

   ```sh
   streamlit run app/main.py
   ```

   Note : Il y a deux fichiers principaux (`main.py` et `MAIN1.py`) dans le répertoire `app`. Assurez-vous d'exécuter le bon.

2. Téléchargez vos données :

   - Allez dans la barre latérale et cliquez sur "Parcourir les fichiers" pour télécharger un fichier PDF, CSV ou texte.

3. Interagissez avec l'assistant :
   - Utilisez la zone de saisie de texte pour poser des questions ou demander une analyse de données.
   - Visualisez les réponses et les visualisations générées par l'assistant.

## Structure des fichiers

```
pdf-rag-main/
├── app/
│   ├── images/
│   ├── utils/
│   │   └── embedding.py
│   ├── main.py
│   └── MAIN1.py
├── architecture/
│   └── [diagram and screenshot files]
├── content/
│   └── [CSV and other data files]
├── data/
│   └── adjusted_close.csv
├── vectorstore/
│   └── db_faiss/
├── [PDF files]
├── README_ENGLISH_VERSION.md
├── README_FRENCH_VERSON.md
└── requirements.txt
```

## Fonctions clés

### `embedding.py`

Le fichier `embedding.py` joue un rôle crucial dans le traitement des documents et la création de la base de données vectorielle. Voici ses principales fonctionnalités :

1. **Chargement de fichiers** :

   - Utilise différents chargeurs (PyPDFLoader, CSVLoader, TextLoader) pour traiter divers types de fichiers (PDF, CSV, TXT).
   - Sauvegarde temporairement les fichiers téléchargés pour le traitement.

2. **Découpage de texte** :

   - Utilise `RecursiveCharacterTextSplitter` pour diviser les documents en morceaux plus petits, facilitant le traitement et la récupération d'informations.

3. **Création d'embeddings** :

   - Utilise `HuggingFaceEmbeddings` pour créer des représentations vectorielles du texte.
   - Emploie le modèle "sentence-transformers/msmarco-distilbert-cos-v5" par défaut.

4. **Stockage vectoriel** :

   - Crée un magasin de vecteurs FAISS à partir des morceaux de documents et de leurs embeddings.

5. **Système de récupération** :
   - Met en place un système de récupération d'ensemble combinant un récupérateur basé sur les vecteurs et un récupérateur BM25.
   - Permet une recherche efficace et précise dans les documents traités.

Cette fonctionnalité est essentielle pour transformer les documents bruts en une forme structurée et recherchable, permettant à l'assistant IA d'accéder rapidement et efficacement aux informations pertinentes.

### Traitement de fichiers :

- `read_csv_file(file_path)` : Lit un fichier CSV et retourne un DataFrame Pandas.
- `process_csv(df)` : Traite un DataFrame et affiche diverses statistiques et visualisations.

### Intégration IA :

- `initqachain(...)` : Initialise la chaîne de questions-réponses avec le modèle de langage sélectionné et les paramètres.
- `create_pandas_agent(df)` : Crée un agent d'interface LangChain pour le DataFrame Pandas.

### Fonctions utilitaires :

- `displayPDF(file_path, page_num)` : Affiche une page PDF sous forme d'image.
- `generate_blog_post(topic, vector_store, k)` : Génère un article de blog basé sur le sujet en utilisant le magasin de vecteurs.

## Contribuer

1. Forkez le dépôt.
2. Créez une nouvelle branche (`git checkout -b branche-fonctionnalite`).
3. Faites vos modifications.
4. Commitez vos changements (`git commit -am 'Ajouter une nouvelle fonctionnalité'`).
5. Poussez vers la branche (`git push origin branche-fonctionnalite`).
6. Créez une nouvelle Pull Request.

## Contact

Pour toute question ou suggestion, veuillez ouvrir une issue sur GitHub ou contacter le mainteneur à leslie.samantha.tientcheu.noumowe@ibm.com.
