import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import random


dataset = pd.read_csv("data/ratings.csv")
print("dataset.shape", dataset.shape)
train_size = 10000
test_size = 3000
batch_size = 100
num_samples = 10000

subset_df = dataset.sample(n=train_size + test_size, random_state=42)

utility_matrix = subset_df.pivot_table(index='userId', columns='movieId', values='rating')

train_matrix = utility_matrix.iloc[: , : int(utility_matrix.shape[1] * 0.7) ]
test_matrix = utility_matrix.iloc[: , int(utility_matrix.shape[1] * 0.7) : ]
print("train_matrix.size : " , train_matrix.shape)
print("train_matrix : \n ", train_matrix)
def processing_data(matrix):
    column_means = matrix.mean()
    matrix1 = matrix.sub(column_means, axis = 1)
    # matrix1 = matrix1.fillna(0)
    return matrix1

train_matrix1 = processing_data(train_matrix)
print("train_matrix1 : \n", train_matrix1)

train_matrix_np = train_matrix1.to_numpy()
np.savetxt('    matrix.csv', train_matrix_np, delimiter=',')
def cosine_base_similarity(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")

    numerator = 0
    denominator1 = 0
    denominator2 = 0
    k = 0
    for (i, j) in zip(v1, v2):
        if not (np.isnan(i) ) and not (np.isnan(j)):
            k+=1
            numerator += i * j
            denominator1 += i**2
            denominator2 += j**2

    print("k = ", k)
    print("numerator : ", numerator)
    print("demoninator ", denominator1, denominator2)
    denominator = np.sqrt(denominator1 * denominator2)
    # return (1.0 * np.dot(v1, v2)) / (1.0 * (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if denominator == 0:
        return 0
    return numerator / denominator
    # Find indices of non-zero elements in both vectors
    # non_zero_indices = np.logical_and(v1 != np.NAN, v2 != np.NAN)
    #
    # # Select only non-zero elements from both vectors
    # v1_filtered = v1[non_zero_indices]
    # v2_filtered = v2[non_zero_indices]
    #
    # # Handle cases where both vectors have only null values
    # if len(v1_filtered) == 0:
    #     return 0  # Consider zero similarity
    #
    # # Calculate cosine similarity using non-zero elements
    # numerator = np.sum(v1_filtered * v2_filtered)
    # denominator = np.sqrt(np.sum(v1_filtered ** 2) * np.sum(v2_filtered ** 2))
    #
    # # Handle potential division by zero (if both vectors have zero variance)
    # if denominator == 0:
    #     return 0  # Consider zero similarity in this case

    # return numerator / denominator

def correlation_based_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

        # Calculate the mean of each vector
    mean1 = np.sum(vec1) / len(vec1)
    mean2 = np.sum(vec2) / len(vec2)

    # Center the vectors by subtracting the mean from each element
    centered_vec1 = vec1 - mean1
    centered_vec2 = vec2 - mean2

    # Calculate the numerator for the correlation coefficient
    numerator = np.sum(centered_vec1 * centered_vec2)

    # Calculate the denominator for the correlation coefficient
    denominator = np.sqrt(np.sum(centered_vec1 ** 2) * np.sum(centered_vec2 ** 2))

    # Handle potential division by zero (if both vectors have zero variance)
    if denominator == 0:
        return 0  # Consider zero similarity in this case

    # Calculate the correlation coefficient
    correlation = numerator / denominator

    return correlation

vector1 = np.array([1, 2, 3])
vector2 = np.array([-4, -8, -10])

similarity = correlation_based_similarity(train_matrix_np[:,0], train_matrix_np[:, 2])
print(f"Correlation-based similarity between vectors: {similarity:.5f}") # return nan, but I want retuen a float number
similarity1 = cosine_base_similarity(train_matrix_np[:, 1],train_matrix_np[:, 2])
print(f"cosine-based similarity between vectors: {similarity1:.5f}") #  return nan
