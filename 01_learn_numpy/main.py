import numpy as np

# Create a 1D array

arr1 = np.array([1, 2, 3, 4, 5])
print("Original array:", arr1)

arr2 = np.array((2, 3))
print("Array with shape (2, 3):", arr2)


# Reshaping arrays

arr3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Original array for reshaping:", arr3)
arr3 = arr3.reshape(3, 3)
print("Reshaped array (3, 3):\n", arr3)


# Zero and One arrays

arr4 = np.zeros((4, 7, 2)) # Creates a 3D array of zeros with np.shape (4, 7, 2)
arr5 = np.ones((3, 5)) # Creates a 2D array of ones with np.shape (3, 5)
print("Array of zeros (4, 7, 2):\n", arr4)
print("Array of ones (3, 5):\n", arr5)


# Array properties

print("Shape of arr4:", arr4.shape)
print("Size of arr4:", arr4.size)
print("Number of dimensions of arr4:", arr4.ndim)
print("Data type of arr4:", arr4.dtype)


# Array indexing and slicing

print("Element at index 1 of arr1:", arr1[1])
print("Element at index 0 of arr3:", arr3[0, 0])
print("Element at index 0 of arr4: ", arr4[0])
print("Slicing arr3 to get first two rows:\n", arr3[:2, :]) # The "," separates the row and column indices. The ":" means all columns, and ":2" means the first two rows.
print("Slicing arr3 to get first two columns:\n", arr3[:, :2]) # The ":" means all rows, and ":2" means the first two columns. (starts from 0 and goes up to but does not include 2)
print("Slicing arr4 to get last two elements of the first row:\n", arr4[0, -2:, :]) # The ":" means all columns, and ":2" means the first two elements of the first row. (starts from 0 and goes up to but does not include 2)


# Flattening arrays

arr6 = arr3.flatten() # Flattens the 2D array into a 1D array
print("Flattened arr3:\n", arr6)


# Array Broadcasting
arr7 = np.array([[1, 2, 3], [4, 5, 6]])
arr8 = np.array([10, 20, 30])
print("Broadcasting arr7 and arr8:\n", arr7 + arr8) # Adds arr8 to each row of arr7

# Boolean Masking and Filtering
arr9 = np.array([1, 2, 3, 4, 5])
arr10 = np.array(["apple", "banana", "cherry", "date", "elderberry"])
mask = arr9 >= 3 # Creates a boolean mask where the condition is True for elements greater than or equal to 3
mask2 = np.char.find(arr10, "a") != -1 # Creates a boolean mask where the condition is True for elements containing the letter "a"
print("Boolean mask for arr9 >= 3:", mask)
print("Filtered arr9 using mask:", arr9[mask])
print("Boolean mask for arr10 containing 'a':", mask2)
print("Filtered arr10 using mask2:", arr10[mask2])


# Exercise
try:
    # 1. Load the tinyShakespeare dataset and the more dataset, and combine them into a single string. Convert it into a list of character codes and turn it into a list of numpy arrays.
    tinyShakespeare = "dataset/input.txt"
    more = "dataset/more.txt"

    with open(tinyShakespeare, "r") as f:
        lines = f.read()

    with open(more, "r") as f:
        lines += f.read()
    
    characters = sorted(set(lines))  # Get the unique characters in the dataset
    char_to_index = {char: idx for idx, char in enumerate(characters)}  # Create a mapping from character to index
    print("Character to index mapping:", char_to_index)
except FileNotFoundError:
    print("Dataset file not found.")

