import ollama
import time
from transformers import AutoTokenizer, AutoModel
import torch

# --- Chargement du dataset depuis "chunk.txt" ---
def load_dataset(file_path="chunk.txt"):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = file.readlines()
    print(f"Loaded {len(dataset)} entries from {file_path}")
    return dataset

# --- Système de calcul d'embeddings et de VectorDB ---
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB = []

def get_embeddings(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate sentence embeddings using a Hugging Face model.
    :param sentences: List of sentences to encode
    :param model_name: Name of the pre-trained model
    :return: Tensor of sentence embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Note : La fonction add_chunk_to_database n'est pas utilisée ici car nous intégrons directement 
# l'embedding dans la boucle ci-dessous.

# --- Boucle pour ajouter chaque chunk dans le VECTOR_DB ---
def build_vector_db(dataset, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    for i, chunk in enumerate(dataset):
        # Ici, on traite chaque chunk individuellement :
        # get_embeddings attend une liste de textes. On passe [chunk]
        embedding = get_embeddings([chunk], model_name=embedding_model)[0]
        VECTOR_DB.append((chunk, embedding))
        print(f"Added chunk {i+1}/{len(dataset)} to the database")

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = get_embeddings([query])[0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        print(embedding.shape)
        print(query_embedding.shape)
        similarity = cosine_similarity(query_embedding.tolist(), embedding.tolist())
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- Pipeline Chatbot utilisant Ollama ---
def run_chatbot(input_query):
    t0 = time.time()
    retrieved_knowledge = retrieve(input_query)
    t1 = time.time()
    
    print("Retrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")
    
    context_chunks = "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    instruction_prompt = f"""You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: 
{context_chunks}"""
    
    t2 = time.time()
    stream = ollama.chat(
      model=LANGUAGE_MODEL,
      messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
      ],
      stream=True,
    )
    t3 = time.time()
    
    print("Chatbot response:")
    for chunk_data in stream:
        print(chunk_data['message']['content'], end='', flush=True)
    print()  # saut de ligne
    
    etape1 = t1 - t0
    etape2 = t2 - t1
    etape3 = t3 - t2
    print("étape1", etape1)
    print("étape2", etape2)
    print("étape3", etape3)
    return stream

# --- Fonction main() qui orchestre tout le pipeline ---
def main(dataset, input_query):
    # Charger le dataset depuis le fichier de chunks
    dataset = load_dataset(dataset)
    
    # Construire le Vector DB avec embeddings
    build_vector_db(dataset)
    
    # Demander à l'utilisateur sa question
    input_query = None
    while input_query == None : 
        input_query = input("Ask me a question: ")
    
    
    # Exécuter le chatbot pour récupérer et afficher la réponse
    stream = run_chatbot(input_query)
    return stream

# --- Exécution directe ---
if __name__ == "__main__":
    stream = main()
