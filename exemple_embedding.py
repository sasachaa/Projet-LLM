
from transformers import AutoTokenizer, AutoModel
import torch

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

if __name__ == "__main__":
    sentences = [
        "Hugging Face models are great!",
        "Transformers are the backbone of modern NLP.",
        "I love using pre-trained models for embeddings."
    ]
   
    embeddings = get_embeddings(sentences)
    print("Embeddings shape:", embeddings.shape)
    print("First embedding:", embeddings[0])