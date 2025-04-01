from pyngrok import ngrok

# Démarrer le tunnel sur le port 8501
public_url = ngrok.connect(8501)
print("Accédez à votre application via :", public_url)
