from nltk.corpus import stopwords
import numpy as np
import re
from pyarabic.araby import tokenize as ar_tokenize
import nltk
from _CustomVectorSimilarity import myCosine_similarity
# nltk.download('punkt')
# nltk.download('stopwords')


def Arabic_remove_stopwords(sentence):
    stop_words = set(stopwords.words('arabic'))
    return [word for word in sentence if word.lower() not in stop_words]


def Arabic_textToVector(value, word_embeddings):
    value = str(value)

    # Tokenize the value into sentences
    # sentences = sent_tokenize(value, language='arabic')
    sentences = ar_tokenize(value)

    # Remove punctuations, numbers, and special characters
    clean_sentences = [re.sub("[^\u0621-\u064A\s]", " ", s) for s in sentences]

    # Convert alphabets to lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    
    # Remove stopwords from the sentences
    clean_sentences = [Arabic_remove_stopwords(r.split()) for r in clean_sentences]

    vector = np.zeros((256,))
    for i in clean_sentences:
        if len(i) != 0:
            word_sum = np.zeros((256,))
            word_count = 0
            for w in i:
                if w in word_embeddings:
                    word_sum += word_embeddings[w]
                    word_count += 1
            if word_count > 0:
                vector += word_sum / (word_count + 0.001)
    return vector


# print("READING Arabic GLOVE NOW....")
# Arabic_word_embeddings = np.load('arabic_word_embeddings.npy', allow_pickle=True).item()
# print("DONE READING Arabic GLOVE !!!!!!")

# p1 = "ذهب الولد الى الحديقة"
# p2 = "الولد ذهب الى الحديقة"


# p1Vec = Arabic_textToVector(p1, word_embeddings)
# p2Vec = Arabic_textToVector(p2, word_embeddings)

# print(p1Vec)
# print(p2Vec)

# print(myCosine_similarity(p1Vec, p2Vec))
