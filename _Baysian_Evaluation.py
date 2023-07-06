def calculate_evaluation_metrics(predicted_labels, true_labels):
    # Calculate true positives, false positives, true negatives, false negatives
    tp = sum((p == "t" and t == "t")
            for p, t in zip(predicted_labels, true_labels))
    fp = sum((p == "t" and t == "f")
            for p, t in zip(predicted_labels, true_labels))
    tn = sum((p == "f" and t == "f")
            for p, t in zip(predicted_labels, true_labels))
    fn = sum((p == "f" and t == "t")
            for p, t in zip(predicted_labels, true_labels))

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate precision
    precision = tp / (tp + fp)

    # Calculate recall
    recall = tp / (tp + fn)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


# predicted_labels = [1, 0, 1, 0, 1, 1, 0, 1]
# true_labels = [1, 1, 1, 0, 1, 1, 1, 1]

# accuracy, precision, recall, f1_score = calculate_evaluation_metrics(
#     predicted_labels, true_labels)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
