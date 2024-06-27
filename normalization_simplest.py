# normalization_simplest.py

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# dataset
dataset = np.array(
    [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]
)

# the MinMaxScaler # the tool that will do the data normalization
scaler = MinMaxScaler()

# fit the scaler to the data and transform it
normalized_dataset = scaler.fit_transform(
    dataset
)

# Print the original and normalized data
print("Original dataset:\n", dataset)
print("Normalized dataset:\n", normalized_dataset)
