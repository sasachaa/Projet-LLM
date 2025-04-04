à installer sur le pod : 
git clone https://github.com/sasachaa/Projet-LLM.git
cd Projet-LLM
apt update
curl -fsSL https://ollama.com/install.sh | sh
apt install ollama

sur un autre terminal ds le pod : ollama serve
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

apt-get install libpango1.0-0
pip install -r requirements.txt

https://7h8c2mvwb59mlg-8501.proxy.runpod.net/

Dans une autre page : 

curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc   |  tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null   && echo "deb https://ngrok-agent.s3.amazonaws.com buster main"   |  tee /etc/apt/sources.list.d/ngrok.list   && apt update   && apt install ngrok	


ngrok config add-authtoken 2uv56qqmgxaDZT25DpiqPrZBXqk_2soktLPAJbN6gMh4sZqkt


ngrok http 8501
