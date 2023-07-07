from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from ToVector import textToVector
from _Baysian_Evaluation import calculate_evaluation_metrics
from sklearn.model_selection import train_test_split
from _CustomVectorSimilarity import myCosine_similarity

# Define the structure of the Bayesian network
model = BayesianModel([('authors', 'source'), ('authors', 'class'),
                       ('source', 'class'), ('title', 'class'), ('summary', 'class')])

# Load the training data
old_data = pd.read_csv(
    '..\\EmbedingsAndData\\training.csv', dtype=str)

word_embeddings = np.load(
    '..\\EmbedingsAndData\\english_word_embeddings.npy', allow_pickle=True).item()
char_embeddings = np.load(
    '..\\EmbedingsAndData\\english_char_embeddings.npy', allow_pickle=True).item()

# make all authors into vectors


def convert_authors_to_vector(authors, word_embeddings, char_embeddings):
    if isinstance(authors, str):
        authors = authors.split(',')  # Split authors by comma
        author_vectors = []
        for author in authors:
            if (author == ""):
                continue
            author_vector = textToVector(
                author.strip(), word_embeddings=word_embeddings, char_embeddings=char_embeddings)
            author_vectors.append(author_vector)
        return author_vectors
    return []


# Convert string features (author, title, source, summary) into vectors
# This step assumes you have already converted the string features into vectors in the training_data.csv file
# Replace the column names and vector conversion code with your specific implementation
# Extract word vectors
old_data['authors'] = old_data['authors'].apply(
    convert_authors_to_vector, word_embeddings=word_embeddings, char_embeddings=char_embeddings)
old_data['title'] = old_data['title'].apply(
    textToVector, word_embeddings=word_embeddings, char_embeddings=char_embeddings)
old_data['source'] = old_data['source'].apply(
    textToVector, word_embeddings=word_embeddings, char_embeddings=char_embeddings)
old_data['summary'] = old_data['summary'].apply(
    textToVector, word_embeddings=word_embeddings, char_embeddings=char_embeddings)

# Split data into data2 (30%) and new data (70%)
data2, data = train_test_split(old_data, test_size=0.3, random_state=42)

# Define a similarity threshold for evidence
similarity_threshold = 0.7

# Predicted values
pridicted_labels = []
actual_labels = []

original_data = data.copy()

# Iterate over each row in data2 and update the corresponding rows in data
for index2, row2 in data2.iterrows():
    # Reset data to its original values
    data = original_data.copy()

    # Convert the evidence vectors for the current row in data2
    if row2['authors']:
        evidence_author = row2[0]
    else:
        evidence_author = row2['authors']
    evidence_title = row2['title']
    evidence_source = row2['source']
    evidence_summary = row2['summary']

    pridicted_labels.append(row2["class"])

    data['authors'] = data.apply(
        lambda row: 'A' if any(myCosine_similarity(
            author, evidence_author) >= similarity_threshold for author in row['authors']) else 'B',
        axis=1
    )
    data['source'] = data.apply(
        lambda row: 'A' if myCosine_similarity(
            row['source'], evidence_source) >= similarity_threshold else 'B',
        axis=1
    )
    data['title'] = data.apply(
        lambda row: 'A' if myCosine_similarity(
            row['title'], evidence_title) >= similarity_threshold else 'B',
        axis=1
    )
    data['summary'] = data.apply(
        lambda row: 'A' if myCosine_similarity(
            row['summary'], evidence_summary) >= similarity_threshold else 'B',
        axis=1
    )

    # Determine the unique categories or levels in the 'authors' and 'source' columns
    unique_authors = data['authors'].unique()
    unique_source = data['source'].unique()
    unique_title = data['title'].unique()
    unique_summary = data['summary'].unique()

    # Set the evidence values to match the unique categories or levels
    evidence_author = 'A' if 'A' in unique_authors else 'B'
    evidence_source = 'A' if 'A' in unique_source else 'B'
    evidence_title = 'A' if 'A' in unique_title else 'B'
    evidence_summary = 'A' if 'A' in unique_summary else 'B'

    # Use the EM algorithm to refine the parameters of the model
    num_categories_author = len(data['authors'].unique())
    num_categories_title = len(data['title'].unique())
    num_categories_source = len(data['source'].unique())
    num_categories_summary = len(data['summary'].unique())

    num_columns = 0
    if num_categories_author > 1:
        num_columns += 1
    if num_categories_title > 1:
        num_columns += 1
    if num_categories_source > 1:
        num_columns += 1
    if num_categories_summary > 1:
        num_columns += 1

    num_categories_class = 2 ** num_columns
    # print("clases")
    # print(num_categories_class)

    be = BayesianEstimator(model, data)
    cpd_author = be.estimate_cpd(
        'authors', prior_type='dirichlet', pseudo_counts=np.ones((num_categories_author, 1)))
    cpd_source = be.estimate_cpd(
        'source', prior_type='dirichlet', pseudo_counts=np.ones((num_categories_source, num_categories_author)))
    cpd_title = be.estimate_cpd(
        'title', prior_type='dirichlet', pseudo_counts=np.ones((num_categories_title, 1)))
    cpd_summary = be.estimate_cpd(
        'summary', prior_type='dirichlet', pseudo_counts=np.ones((num_categories_summary, 1)))
    cpd_class = be.estimate_cpd(
        'class', prior_type='dirichlet', pseudo_counts=np.ones((2, num_categories_class)))

    print(cpd_author)
    print(cpd_title)
    print(cpd_source)
    print(cpd_summary)
    print(cpd_class)

    model.add_cpds(cpd_author, cpd_title, cpd_source, cpd_summary, cpd_class)

    # Create an inference engine to perform inference on the network
    inference = VariableElimination(model)

    # Make predictions for new data using the learned model and filtered evidence
    query = inference.query(['class'], evidence={
                            'authors': evidence_author, 'title': evidence_title,
                            'source': evidence_source, 'summary': evidence_summary}, joint=False)

    print(query['class'])

    prob1 = query['class'].values[0]
    prob2 = query['class'].values[1]

    if prob1 >= prob2:
        actual_labels.append("f")
    else:
        actual_labels.append("t")


print(pridicted_labels)
print(actual_labels)

accuracy, precision, recall, f1_score = calculate_evaluation_metrics(
    predicted_labels=pridicted_labels, true_labels=actual_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
