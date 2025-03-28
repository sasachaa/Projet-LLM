à installer sur le pod : 
git clone https://github.com/sasachaa/Projet-LLM.git
cd Projet-LLM
apt update
curl -fsSL https://ollama.com/install.sh | sh
apt install ollama
pip install ollama
pip install langchain
pip install transformers
sur un autre terminal ds le pod : ollama serve
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
pip install httpx==0.28
apt-get install libpango1.0-0
pip install -r requirements.txt

https://7h8c2mvwb59mlg-8501.proxy.runpod.net/
