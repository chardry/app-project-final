# Importation des packages
import random
import time
from pathlib import Path

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
import streamlit as st
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Définition des éléments
fichier = Path.home() / "PycharmProjects" / "app-project-final" / "lions.txt"

lignes = fichier.read_text()
lignes = lignes.lower()

tokens_phrase = nltk.sent_tokenize(lignes)
tokens_mots = nltk.word_tokenize(lignes)

tokenizer = RegexpTokenizer(r'\w+')


def tokenisation(text):
    tokens1 = []
    tokens1 += tokenizer.tokenize(text.lower())
    return tokens1


salutations = ("salut", "bonjour", "hello", "bonjour, comment ça va ?")
rep_salutations = ("Salut", "Bonjour", "Hello", "Bonjour, ça me fait plaisir de répondre à vos questions")


def saluer(phrase):
    if phrase != None:
        for word in phrase.split():
            if word in salutations:
                return random.choice(rep_salutations)


mots = stopwords.words('french')


def repondre(utilisateur):
    chatbot_rep = ''

    if utilisateur != None:
        tokens_phrase.append(utilisateur)
        TfidfVec = TfidfVectorizer(tokenizer=tokenisation, stop_words=mots)  # english
        tfidf = TfidfVec.fit_transform(tokens_phrase)
        vals = cosine_similarity(tfidf[-1], tfidf)  # type: ignore
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if (req_tfidf == 0):
            chatbot_rep = chatbot_rep + "Je ne comprend pas. Pouvez vous reformulez votre question ?"
            return chatbot_rep
        else:
            chatbot_rep = chatbot_rep + tokens_phrase[idx]
            return chatbot_rep


st.title("ChatBot sur les Lions")
st.markdown("Répondre à vos questions concernant les lions. Pour quitter << au revoir >>")

# Initialiser l'historique des discussions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de discussion de l'historique lors de la réexécution de l'application
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Réagir aux entrées de l'utilisateur
if prompt := st.chat_input("Veuillez saisir ... ?"):
    # Afficher le message de l'utilisateur dans le conteneur de messages de discussion
    with st.chat_message("user"):
        st.markdown(prompt)
    # Ajouter un message utilisateur à l'historique des discussions
    st.session_state.messages.append({"role": "user", "content": prompt})


# Émulateur de réponse en streaming
def response_generator():
    response = ""

    utilisateur = str(prompt)
    utilisateur = utilisateur.lower()

    if (utilisateur != 'au revoir'):
        if (utilisateur == 'merci'):
            response = "Bienvenue, ma mission est de donner des réponses à vos questions"
        else:
            if (saluer(utilisateur) != None):
                response = saluer(utilisateur)
            else:
                response = repondre(utilisateur)
                tokens_phrase.remove(utilisateur)
    else:
        response = "Au revoir et n'oubliez pas que le Lion est le roi de la jungle"

    for word in response.split():  # type: ignore
        yield word + " "
        time.sleep(0.05)


# Afficher la réponse de l'assistant dans le conteneur de messages de discussion
with st.chat_message("assistant"):
    response = st.write_stream(response_generator)

# Ajouter la réponse de l'assistant à l'historique des discussions
st.session_state.messages.append({"role": "assistant", "content": response})