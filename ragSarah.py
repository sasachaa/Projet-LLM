import ollama
import time
from transformers import AutoTokenizer, AutoModel
import torch
# Load the dataset



dataset = []
with open('chunk.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')



# Implement the retrieval system

LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def get_embeddings(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate sentence embeddings using a Hugging Face model.
   
    :param sentences: List of sentences to encode
    :param model_name: Name of the pre-trained model
    :return: Tensor of sentence embeddings
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
   
    # Tokenize input texts
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
   
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
   
    # Extract embeddings (mean pooling over the last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1)
   
    return embeddings

sentences = dataset

## changer add_chunk_to_database dans la oucle car notre pb c'est qu'il prend seulement dans l'embedding la première phrase donc il faudrait l'intégrer dans la boucle for
def add_chunk_to_database(chunk):
  for i, chunk in enumerate(dataset):
    embedding = get_embeddings(sentences)[i]
    VECTOR_DB.append((chunk, embedding))
  
for i, chunk in enumerate(dataset):
  #add_chunk_to_database(chunk)
  embedding = get_embeddings(chunk)[0]
  VECTOR_DB.append((chunk, embedding))
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
  query_embedding = get_embeddings(query)[0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    print(embedding.shape)
    print(query_embedding.shape)
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



# Chatbot

input_query = input('Ask me a question: ')
t0 = time.time()
retrieved_knowledge = retrieve(input_query)


print('Retrieved knowledge:')
t1 = time.time()
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

context_chunks = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
instruction_prompt = f'''You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: 
{context_chunks}'''
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

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

etape1 = t1 - t0
etape2 = t2 - t1
etape3 = t3 - t2
print("étape1",etape1)
print("étape2",etape2)
print("étape3",etape3)