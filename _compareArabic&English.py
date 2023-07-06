import numpy as np
from sklearn.decomposition import PCA
from ToVector import textToVector
from _ArabicGloveTest import Arabic_textToVector
from _CustomVectorSimilarity import myCosine_similarity

print("READING Arabic GLOVE NOW....")
# Loading word embeddings
Arabic_word_embeddings = np.load(
    '..\\EmbedingsAndData\\arabic_word_embeddings.npy', allow_pickle=True).item()
print("DONE READING Arabic GLOVE !!!!!!")

print("READING English GLOVE NOW....")
# Loading word embeddings
word_embeddings = np.load(
    '..\\EmbedingsAndData\\english_word_embeddings.npy', allow_pickle=True).item()
print("DONE READING English GLOVE !!!!!!")

# *********************************
Ap = "رجل"
ApVec = Arabic_textToVector(Ap, word_embeddings=Arabic_word_embeddings)

Ep = "man"
EpVec = textToVector(Ep, word_embeddings=word_embeddings)

print(ApVec)
print(EpVec)

# make english shorter: 256
EpVec = EpVec[:len(ApVec)]
print(myCosine_similarity(ApVec, EpVec))

# make arabic longer: 300
ApVec = np.pad(ApVec, (0, len(EpVec) - len(ApVec)), mode='constant')
print(myCosine_similarity(ApVec, EpVec))
