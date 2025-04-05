==================== READ ME ====================
Auteurs : 
Chaabouni Sarah
Troubat Victoria

Ce chatbot prend en entrée un pdf de votre choix et vou permet de poser les questions de votre choix sur ce pdf. 
Il est paramétré pour ne pouvoir regarder que des pdf de moins de 20 pages.
Il vous faudra un GPU.
Nous avons utilisé RunPod pour accéder à un GPU via notre terminal.

Après avoir ouvert un pod sur run pod dans "community cloud" :
copiez la clé ssh de votre pod

Dans votre terminal :
allez dans le dossier .ssh 
collez la clé ssh (cela devrait ouvrir votre pod)

Installer sur le pod ces lignes de commandes : 

git clone https://github.com/sasachaa/Projet-LLM.git
cd Projet-LLM
apt update
curl -fsSL https://ollama.com/install.sh | sh
apt install ollama

Ouvrez une deuxième page de terminal et entrez : ollama serve
gardez cette page ouverte 

Retournez sur la page 1 du terminal et entrez ces commandes : 

ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
apt-get install libpango1.0-0
pip install -r requirements.txt

Ouvrez une troisième page de terminal et entrez ces commandes

curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc   |  tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null   && echo "deb https://ngrok-agent.s3.amazonaws.com buster main"   |  tee /etc/apt/sources.list.d/ngrok.list   && apt update   && apt install ngrok	

ngrok config add-authtoken 2uv56qqmgxaDZT25DpiqPrZBXqk_2soktLPAJbN6gMh4sZqkt

ngrok http 8501

retournez sur le premier terminal :
cd application
streamlit run main.py 

pour accéder à l'application retournez sur le terminal 3 et cliquez sur le lien "forwarding"
