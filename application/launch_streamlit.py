from pyngrok import ngrok

# Démarrer le tunnel sur le port 8501
ngrok.set_auth_token("2uv56qqmgxaDZT25DpiqPrZBXqk_2soktLPAJbN6gMh4sZqkt")
public_url = ngrok.connect(8501)
print("Accédez à votre application via :", public_url)
