# standardization_simplest.py

from sklearn.preprocessing import StandardScaler
import numpy as np

# the dataset
dataset = np.array(
    [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]
)

# the StandardScaler # the tool that will do the standardization
scaler = StandardScaler()

# fit the scaler to the data and transform it
standardized_dataset = scaler.fit_transform(
    dataset
)

print("Original data:\n", dataset)
print("Standardized data:\n", standardized_dataset)
