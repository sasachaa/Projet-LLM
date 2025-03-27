# retrieval_chat.py
"""
Script unique pour :
1) Charger un fichier de chunks .txt (un chunk par ligne),
2) Construire un VectorDB (embeddings HuggingFace -> mean pooling),
3) Poser une question utilisateur,
4) Récupérer les chunks les plus pertinents,
5) Appeler Ollama (modèle local) pour répondre,
6) Afficher la réponse en temps réel,
7) Afficher des mesures de temps (étapes 1,2,3).

Fonction main(chunk_file, question, embedding_model, language_model) pour 
tout enchaîner. 
"""

import time
import torch
import ollama
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Paramètres par défaut
# -------------------------
DEFAULT_CHUNK_FILE = "chunk_copy.txt"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


# -------------------------
# Fonctions pour embeddings & retrieval
# -------------------------
def get_embeddings(sentences, model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Genère des embeddings pour chaque phrase d'une liste, 
    via un modèle Transformers Hugging Face (mean pooling).
    Retourne un tenseur PyTorch (N, hidden_dim).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def cosine_similarity_torch(a, b):
    """cosine similarity entre deux tenseurs 1D PyTorch."""
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

# -------------------------
# Classe / Structure pour VectorDB
# -------------------------
class VectorDB:
    """
    Stocke une liste de (chunk_text, embedding_tensor).
    """
    def __init__(self):
        self.db = []  # liste de tuples (text, tensor)

    def add_chunk(self, chunk_text, embedding):
        self.db.append((chunk_text, embedding))

    def retrieve(self, query, top_n=3, embedding_model=DEFAULT_EMBEDDING_MODEL):
        """
        - Calcule l'embedding du query (liste d'une seule phrase).
        - Compare la similarité cosinus avec chaque chunk.
        - Retourne les (chunk, similarity) triés desc, top_n.
        """
        query_embedding = get_embeddings([query], model_name=embedding_model)[0]
        scored = []
        for (chunk_text, emb) in self.db:
            sim = cosine_similarity_torch(query_embedding, emb)
            scored.append((chunk_text, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

# -------------------------
# Pipeline complet
# -------------------------
def main(
    chunk_file=DEFAULT_CHUNK_FILE,
    question=None,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    language_model=DEFAULT_LANGUAGE_MODEL
):
    """
    1) Charge le fichier chunk_file (un chunk par ligne) 
    2) Calcule embeddings pour chaque chunk -> VectorDB
    3) Si question=None, on la demande via input()
    4) On récupère les top_n=3 plus pertinents
    5) On construit un prompt -> Ollama
    6) On stream la réponse
    7) Affiche les temps d'exécution d'étapes
    """
    # 0) Charger le dataset (fichier .txt)
    dataset_path = Path(chunk_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {chunk_file}")

    with open(dataset_path, "r") as f:
        dataset = f.readlines()
    print(f"Loaded {len(dataset)} entries from {dataset_path}")

    # 1) Construction VectorDB
    t0 = time.time()
    vector_db = VectorDB()

    # On ajoute chaque chunk dans la db
    # ATTENTION : dans le code initial, add_chunk_to_database prenait 
    # "sentences=[dataset]" -> un bug potentiel. On veut 
    # chaque chunk individuellement => get_embeddings([chunk])
    for i, chunk_text in enumerate(dataset):
        chunk_text = chunk_text.strip()
        emb = get_embeddings([chunk_text], model_name=embedding_model)[0]
        vector_db.add_chunk(chunk_text, emb)
        print(f"Added chunk {i+1}/{len(dataset)} to the database")
    t1 = time.time()

    # 2) Si question n'est pas fournie, on la demande
    if not question:
        question = input('Ask me a question: ')

    # 3) retrieve top_n=3
    retrieved_knowledge = vector_db.retrieve(question, top_n=3, embedding_model=embedding_model)
    t2 = time.time()

    print('Retrieved knowledge:')
    for chunk_text, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk_text}')

    # 4) Construction prompt
    context_chunks = '\n'.join([f' - {c}' for c, _ in retrieved_knowledge])
    instruction_prompt = (
        "You are a helpful chatbot. Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n" + context_chunks
    )

    # 5) Appel Ollama (streaming)
    stream = ollama.chat(
      model=language_model,
      messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': question},
      ],
      stream=True,
    )
    # 6) Affichage
    print('Chatbot response:')
    for chunk_data in stream:
        print(chunk_data['message']['content'], end='', flush=True)

    print()  # saut de ligne
    t3 = time.time()

    # 7) Mesures temps
    etape1 = t1 - t0  # Construction DB
    etape2 = t2 - t1  # retrieve
    etape3 = t3 - t2  # chat
    print("étape1 (build DB)  :", etape1, "s")
    print("étape2 (retrieve) :", etape2, "s")
    print("étape3 (chat)     :", etape3, "s")


# -------------------------
# Execution directe
# -------------------------
if __name__ == "__main__":
    # Ex : chunk_file="chunk_copy.txt"
    main()  # on peut préciser main(chunk_file="chunk_copy.txt", question="Hello?")
