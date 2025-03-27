from langchain.text_splitter import CharacterTextSplitter
import numpy as np
text_splitter = CharacterTextSplitter(
    separator="." ,      # on coupe prioritairement sur les sauts de ligne
    chunk_size=400,      # taille max (en caractères)
    chunk_overlap=0,      
    # éventuellement un paramètre length_function, si on veut 
    # calculer la taille autrement, mais la valeur par défaut (len) suffit
)

with open("cat-facts.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunk = text_splitter.split_text(content)

with open('chunk.txt2', 'w', encoding='utf-8') as f:
    f.write('\n'.join(chunk))
