import numpy as np

# Path to the GloVe embeddings text file
# glove_file = 'C:\\Users\\laith\\Desktop\\glove.840B.300d-char.txt'

# # # Read the GloVe embeddings from the text file
# word_embeddings = {}
# f = open(glove_file, encoding='utf-8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     word_embeddings[word] = coefs
# f.close()

# # # Convert the list of embeddings to a NumPy array
# glove_embeddings = np.array(word_embeddings)

# # # Save the GloVe embeddings to a file
# np.save('english_char_embeddings.npy', glove_embeddings)

# *************************************
# reading
word_embeddings = np.load(
    '..\\EmbedingsAndData\\english_char_embeddings.npy', allow_pickle=True)
