from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ToVector import textToVector

# Define the structure of the Bayesian network
model = BayesianModel([('authors', 'source'), ('authors', 'class'), ('source', 'class')])

# Load the training data
data = pd.read_csv('C:\\Users\\laith\\Desktop\\baysian project\\DemoTraining.csv')

word_embeddings = {}
f = open('glove.6B\glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

# Convert string features (author, title, source, summary) into vectors
# This step assumes you have already converted the string features into vectors in the training_data.csv file
# Replace the column names and vector conversion code with your specific implementation
# Extract word vectors
data['authors'] = data['authors'].apply(textToVector, word_embeddings=word_embeddings)
data['source'] = data['source'].apply(textToVector, word_embeddings=word_embeddings)

print(data['authors'].get(0))


# Define a similarity threshold for evidence
similarity_threshold = 0.7

#/////////////////testdata
data2 = pd.read_csv('C:\\Users\\laith\\Desktop\\baysian project\\DemoEvidance.csv')

# # Convert the evidence vectors
evidence_author = data2['authors'].apply(textToVector, word_embeddings=word_embeddings)
evidence_source = data2['source'].apply(textToVector, word_embeddings=word_embeddings)
evidence_author = evidence_author[0]
evidence_source = evidence_source[0]

def myCosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# print(myCosine_similarity(data['authors'][0],evidence_author))

data['authors'] = data.apply(
    lambda row: 'A' if myCosine_similarity(row['authors'], evidence_author) >= similarity_threshold else 'B',
    axis=1
)
data['source'] = data.apply(
    lambda row: 'A' if myCosine_similarity(row['source'], evidence_source) >= similarity_threshold else 'B',
    axis=1
)

# Determine the unique categories or levels in the 'authors' and 'source' columns
unique_authors = data['authors'].unique()
unique_source = data['source'].unique()

# Set the evidence values to match the unique categories or levels
evidence_author = unique_authors[0]  # Choose the first category or level
evidence_source = unique_source[0]  # Choose the first category or level

# Use the EM algorithm to refine the parameters of the model
num_categories_author = len(data['authors'].unique())
num_categories_source = len(data['source'].unique())
num_categories_class = len(data['class'].unique())

be = BayesianEstimator(model, data)
cpd_author = be.estimate_cpd('authors', prior_type='dirichlet', pseudo_counts=np.ones((2,1)))
cpd_source = be.estimate_cpd('source', prior_type='dirichlet', pseudo_counts=np.ones((1,2)))
cpd_class = be.estimate_cpd('class', prior_type='dirichlet', pseudo_counts=np.ones((2,2)))

print(cpd_author)
print(cpd_source)
print(cpd_class)

model.add_cpds(cpd_author, cpd_source, cpd_class)

# Create an inference engine to perform inference on the network
inference = VariableElimination(model)

# Make predictions for new data using the learned model and filtered evidence
query = inference.query(['class'], evidence={'authors': evidence_author, 'source': evidence_source}, joint=False)

print(query['class'])
