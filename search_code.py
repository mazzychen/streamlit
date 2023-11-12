import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle 


# Compute Cosine Similarity
def cosine_similarity(x,y):

    x_arr = np.array(x)
    y_arr = np.array(y)

    return np.dot(x_arr,y_arr)/(np.linalg.norm(x_arr)*np.linalg.norm(y_arr))

# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path="./glove.6B.50d.txt"):
    embeddings_dict = {}
    with open(glove_path,"r") as f:
        for line in f:
            word = line.split(" ")[0]
            embedding = np.array([float(val) for val in line.split(" ")[1:]])
            embeddings_dict[word] = embedding
        # embeddings_dict = pickle.load(f)
    
    # print(embeddings_dict["cat"])
    # print(len(embeddings_dict["cat"]))
    return embeddings_dict

# Get Averaged Glove Embedding of a sentence
def averaged_glove_embeddings(sentence, embeddings_dict):
    words = sentence.split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0

    for word in words:
        glove_embedding += embeddings_dict[word]
        count_words += 1
        
    return glove_embedding/count_words

# Load glove embeddings
glove_embeddings = load_glove_embeddings()

# Gold standard words to search from
gold_words = ["flower","mountain","tree","car","building"]

# Text Search
st.title("Search Based Retrieval Demo")
st.subheader("Pass in an input word or even a sentence (e.g. jasmine or mount adams)")
text_search = st.text_input("", value="")


# Find closest word to an input word
if text_search:
    input_embedding = averaged_glove_embeddings(text_search, glove_embeddings)
    cosine_sim = {}
    for index in range(len(gold_words)):
        cosine_sim[index] = cosine_similarity(input_embedding, glove_embeddings[gold_words[index]])

    # Sort the cosine similarities
    sorted_cosine_sim = sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)

    st.write("(My search uses glove embeddings)")
    st.write("Closest picture I have between flower, mountain, tree, car and building for your input is: ")
    path = f"{gold_words[sorted_cosine_sim[0][0]]}.jpeg"
    st.image(path)
    st.write("")

