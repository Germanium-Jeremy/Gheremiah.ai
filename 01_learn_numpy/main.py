import numpy as np

# ============================================================================
# STAGE 01 — Learn NumPy: the building blocks used by every later stage.
# ----------------------------------------------------------------------------
# Goal of this file: get comfortable with arrays (the container that holds all
# token data), then apply them to a real NLP task: a CHARACTER-LEVEL BIGRAM
# language model. A bigram model predicts the next character using ONLY the
# single character before it. It is "the dumbest model that can generate text"
# (Karpathy) — a counting baseline that later neural stages try to beat.
# Pipeline here: text -> char indices -> bigram count matrix -> row-normalize
# -> probability matrix -> sample next char. Every later stage reuses this
# pipeline with bigger models, so understanding it here pays off everywhere.
# ============================================================================


# Create a 1D array

# arr1 is the base array used for the indexing demo below. Shape (5,).
# Note: np.array defaults to integer dtype (int64 on most platforms) here.
arr1 = np.array([1, 2, 3, 4, 5])
print("Original array:", arr1)

# arr2 shows a tuple also works as input; shape (2,). Just syntax practice.
arr2 = np.array((2, 3))
print("Array:", arr2)


# Reshaping arrays

# arr3 demonstrates reshape: same 9 elements, reinterpreted as (3, 3).
# reshape() returns a VIEW when possible, not a copy — so renaming arr3 to the
# reshaped version is cheap, but mutating it can affect other views of the
# same memory. reshape needs the total size to match: 3*3 == 9.
arr3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Original array for reshaping:", arr3)
arr3 = arr3.reshape(3, 3)
print("Reshaped array (3, 3):\n", arr3)


# Zero and One arrays

# arr4: 3D zeros, shape (4, 7, 2) = 4 * 7 * 2 = 56 elements, all 0.
# Useful for pre-allocating buffers — e.g. the bigram_counts matrix below is
# exactly this pattern (2D), pre-filled with zeros then counted into.
arr4 = np.zeros((4, 7, 2)) # Creates a 3D array of zeros with np.shape (4, 7, 2)
arr5 = np.ones((3, 5)) # Creates a 2D array of ones with np.shape (3, 5)
print("Array of zeros (4, 7, 2):\n", arr4)
print("Array of ones (3, 5):\n", arr5)


# Array properties

# .shape = size per dimension, .size = total element count (4*7*2 = 56),
# .ndim = number of axes (3 here), .dtype = element type (float64 for zeros).
# In LLM code you constantly assert/check shapes — these are the four tools.
print("Shape of arr4:", arr4.shape)
print("Size of arr4:", arr4.size)
print("Number of dimensions of arr4:", arr4.ndim)
print("Data type of arr4:", arr4.dtype)


# Array indexing and slicing

# arr1[1] -> basic 0-based indexing (the "2" of the original list).
print("Element at index 1 of arr1:", arr1[1])
# arr3[0, 0] -> 2D indexing with a comma: row 0, column 0.
# Equivalent alternative: arr3[0][0] (chained) — but the comma form is one
# memory access and is the idiomatic NumPy style.
print("Element at index 0 of arr3:", arr3[0, 0])
# arr4[0] -> indexing one axis gives a view of shape (7, 2); not a copy.
print("Element at index 0 of arr4: ", arr4[0])
# Slicing arr3 to get first two rows:
# rows :2 (start up to but NOT including 2), columns : (all).
print("Slicing arr3 to get first two rows:\n", arr3[:2, :]) # The "," separates the row and column indices. The ":" means all columns, and ":2" means the first two rows.
# Mirror image: all rows, first two columns. Slices are views, not copies —
# crucial for the get_batch windows in stages 05/06 (they slice, not copy).
print("Slicing arr3 to get first two columns:\n", arr3[:, :2]) # The ":" means all rows, and ":2" means the first two columns. (starts from 0 and goes up to but does not include 2)
# Negative indexing: -2: = last two elements of axis 1 (0-based counting
# from the end). Used a lot for "last N tokens" style context cropping.
print("Slicing arr4 to get last two elements of the first row:\n", arr4[0, -2:, :]) # The ":" means all columns, and ":2" means the first two elements of the first row. (starts from 0 and goes up to but does not include 2)


# Flattening arrays

# flatten() returns a COPY in row-major (C) order; ravel() would return a
# view when possible. Note: the MLP stage (04) does the same flattening when
# it turns (batch, seq, vocab) one-hot tensors into flat input vectors.
arr6 = arr3.flatten() # Flattens any multidimensional array into a 1D array
print("Flattened arr3:\n", arr6)


# Array Broadcasting
# Broadcasting: NumPy stretches the (3,) vector across rows instead of
# copying it. Rule: dimensions compare right-to-left and must be equal or 1.
# This is how row normalization works below and how bias vectors are added
# to batches in PyTorch — same mechanics.
arr7 = np.array([[1, 2, 3], [4, 5, 6]])
arr8 = np.array([10, 20, 30])
print("Broadcasting arr7 and arr8:\n", arr7 + arr8) # Adds arr8 to each row of arr7

# Boolean Masking and Filtering
# A boolean mask is an array of True/False of the same length; indexing with
# it keeps only the True positions. Pattern used everywhere for filtering.
arr9 = np.array([1, 2, 3, 4, 5])
arr10 = np.array(["apple", "banana", "cherry", "date", "elderberry"])
# Comparison creates the mask element-wise (no loop needed).
mask = arr9 >= 3 # Creates a boolean mask where the condition is True for elements greater than or equal to 3
# np.char.find returns -1 when not found, so != -1 means "contains 'a'".
mask2 = np.char.find(arr10, "a") != -1 # Creates a boolean mask where the condition is True for elements containing the letter "a"
print("Boolean mask for arr9 >= 3:", mask)
print("Filtered arr9 using mask:", arr9[mask])
print("Boolean mask for arr10 containing 'a':", mask2)
print("Filtered arr10 using mask2:", arr10[mask2])


# Exercise
# 1. Load the tinyShakespeare dataset and the more dataset, and combine them into a single string. Convert it into a list of character codes and turn it into a list of numpy arrays.
# This is the text->tokens pipeline every stage reuses:
#   raw string -> set of unique chars -> sorted vocab -> char->index map ->
#   integer sequence. sorted() makes the vocab deterministic/reproducible.
try:
    # Paths relative to the project root (run the script from there).
    tinyShakespeare = "dataset/input.txt"
    more = "dataset/more.txt"

    with open(tinyShakespeare, "r") as f:
        lines = f.read()

    with open(more, "r") as f:
        lines += f.read()
    
    # characters = the vocabulary: every distinct character exactly once, in
    # sorted order. Its length is the vocab_size that decides the size of the
    # bigram matrix (and of every weight matrix in later stages).
    characters = sorted(set(lines))  # Get the unique characters in the dataset
    # char_to_index: the tokenizer. Text can't be fed to math, so each char
    # becomes its integer index. Any char NOT in this dict would KeyError —
    # that's why the model can only ever output vocabulary characters.
    char_to_index = {char: idx for idx, char in enumerate(characters)}  # Create a mapping from character to index
    # Encode the whole corpus as a flat list of integer token ids.
    char_codes = [char_to_index[char] for char in lines]  # Convert the characters in the dataset to their corresponding indices
    char_codes_array = np.array(char_codes)  # Convert to NumPy array
    # print("Character codes array:", char_codes_array[-10:])
except FileNotFoundError:
    print("Dataset file not found.")

# 2: Create a bigram count matrix
# vocab_size drives EVERYTHING: the matrix is (vocab_size, vocab_size), and in
# later stages it sets input/output dimensions of the neural nets. Bigger
# vocab = bigger matrices = more parameters to learn.
vocab_size = len(characters)  # Number of unique characters
# Pre-allocate an int count matrix. Entry [i, j] will hold "how many times did
# character i get followed by character j in the corpus".
bigram_counts = np.zeros((vocab_size, vocab_size), dtype=int) # Create a 2D array of zeros with shape (vocab_size, vocab_size) to hold the bigram counts

# Count bigrams
# zip(a[:-1], a[1:]) pairs each token with its successor: this IS the
# bigram dataset. char_codes_array[:-1] = every "current" char,
# char_codes_array[1:] = every "next" char. This shifted-pair trick is the
# core of all next-token training in stages 05/06 (see get_batch there).
for (i, j) in zip(char_codes_array[:-1], char_codes_array[1:]): # Iterate through the character codes array, taking pairs of consecutive characters (bigrams)
    bigram_counts[i, j] += 1
    # print(f"Bigram ({characters[i]}, {characters[j]}) count: {bigram_counts[i, j]}")  # Print the count of each bigram as it is counted


print("Bigram count matrix shape:", bigram_counts.shape)  # Should be (65, 65)


# 3: Normalize the rows to create a probability distribution
# Dividing each row by its sum turns raw counts into P(next | current):
# row i becomes a probability distribution over the next character.
# keepdims=True keeps shape (vocab, 1) so broadcasting divides each row by
# its own sum (see the broadcasting demo above).
row_sums = bigram_counts.sum(axis=1, keepdims=True)  # Sum of each row
probability_matrix = bigram_counts / row_sums  # Normalize

# Handle divisions by zero (if any row sum is zero)
# A character that never appears as a "current" char gives row_sums = 0 ->
# 0/0 = NaN. NaN would break np.random.choice(p=...) (it validates that p
# sums to 1), so NaNs are replaced with 0 probability.
probability_matrix = np.nan_to_num(probability_matrix)  # Replace NaN with 0

print("Normalized probability matrix shape:", probability_matrix.shape)  # Should be (65, 65)


# 4: Sample the next character index based on the current character index
# Sampling (np.random.choice) instead of always taking the most likely char
# (argmax) is what makes generated text varied. Greedy/argmax generation is
# deterministic and loops forever once a context repeats — same reason the
# later stages use torch.multinomial with a temperature.
def sample_next_char(current_char_index):
    """Sample the next character index based on the current character index."""
    # Row lookup = P(next | current). Each call is one "step" of generation.
    probabilities = probability_matrix[current_char_index]
    # p must sum to 1 (guaranteed by normalization + NaN cleanup).
    next_char_index = np.random.choice(range(vocab_size), p=probabilities)
    return next_char_index

# Test the sampling function
# Index 0 = first char of the sorted vocab (a '\n' for this dataset).
current_index = 0  # Example: starting with the first character
next_index = sample_next_char(current_index)
print("Next character index sampled:", next_index)
print("Next character:", characters[next_index])
