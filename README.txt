à installer sur le pod : 
apt update
curl -fsSL https://ollama.com/install.sh | sh
apt install ollama
pip install ollama
pip install langchain
pip install transformers
sur un autre terminal ds le pod : ollama serve
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
apt-get install unzip
unzip ngrok-stable-linux-amd64.zip
chmod +x ngrok
ls -l ngrok

