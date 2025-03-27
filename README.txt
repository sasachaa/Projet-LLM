à installer sur le pod : 
apt update
curl -fsSL https://ollama.com/install.sh | sh
apt install ollama
pip install ollama
pip install langchain
pip install transformers

ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
