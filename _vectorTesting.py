import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')  # one-time execution
# nltk.download('stopwords')
from nltk.corpus import stopwords
from ToVector import textToVector
from _CustomVectorSimilarity import myCosine_similarity

word_embeddings = np.load(
    '..\\EmbedingsAndData\\english_word_embeddings.npy', allow_pickle=True).item()


# Example values
p1 = "Wendy Gittleson"
p2 = "Wendy Gittleson,Tony Elliott"

p1Vec = textToVector(p1, word_embeddings)
p2Vec = textToVector(p2, word_embeddings)


print(myCosine_similarity(p1Vec, p2Vec))
