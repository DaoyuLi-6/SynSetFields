from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import torch


# Initialize BERT tokenizer and model
def initialize_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # Use the [CLS] token representation as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def calculate_average_similarity_per_line(data, tokenizer, model):
    results = []

    for line in data:
        words = [item.split('||')[0] for item in line.split(',')]
        embeddings = np.array([compute_embedding(word, tokenizer, model) for word in words])

        # Calculate pairwise cosine similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Compute average similarity for the line
        avg_similarity = np.mean(similarities)
        results.append(avg_similarity)

    # Calculate overall statistics
    overall_avg_similarity = np.mean(results)
    std_dev_similarity = np.std(results)

    return results, overall_avg_similarity, std_dev_similarity


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]


def main(file_path):
    data = read_data_from_file(file_path)
    tokenizer, model = initialize_bert()
    results, overall_avg_similarity, std_dev_similarity = calculate_average_similarity_per_line(data, tokenizer, model)

    print("Per-line average similarities:", results)
    print("Overall average similarity:", overall_avg_similarity)
    print("Standard deviation of similarities:", std_dev_similarity)

    return results, overall_avg_similarity, std_dev_similarity


# Example usage
if __name__ == "__main__":
    file_path = "results/NYT_predicted_sets.txt"  # Replace with your file path
    main(file_path)
