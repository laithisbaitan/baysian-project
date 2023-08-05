import pandas as pd
import numpy as np
import networkx as nx
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
# nltk.download('punkt') # one time execution
# nltk.download('stopwords')


# Extract word vectors
word_embeddings = np.load(
    '..\\EmbedingsAndData\\english_word_embeddings.npy', allow_pickle=True).item()


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    return [word for word in sentence if word.lower() not in stop_words]

# Define your function to be applied to the column


def TextToSummery(value):
    # Modify the value as needed
    sentences = sent_tokenize(value)

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = [sum([word_embeddings.get(word, np.zeros((300,))) for word in sentence]) / (
        len(sentence) + 0.001) if len(sentence) != 0 else np.zeros((300,)) for sentence in clean_sentences]

    # print(sentence_vectors)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 300),
                                                  sentence_vectors[j].reshape(1, 300))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    final_summary = ranked_sentences[0][1]  # Extract the top-ranked sentence

    return final_summary
