import numpy as np

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path
        self.more_path = more_path
        self.characters = []
        self.char_to_index = {}
        self.char_codes_array = None
        self.probability_matrix = None

    def load_data(self):
        """Load the datasets and create character codes."""
        try:
            with open(self.tiny_shakespeare_path, "r") as f:
                lines = f.read()

            with open(self.more_path, "r") as f:
                lines += f.read()

            self.characters = sorted(set(lines))  # Get unique characters
            self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}  # Mapping from char to index
            char_codes = [self.char_to_index[char] for char in lines]  # Convert to indices
            self.char_codes_array = np.array(char_codes)  # Convert to NumPy array
        except FileNotFoundError:
            print("Dataset file not found.")

    def create_bigram_matrix(self):
        """Create a bigram count matrix and normalize it."""
        vocab_size = len(self.characters)  # Number of unique characters
        bigram_counts = np.zeros((vocab_size, vocab_size), dtype=int)  # Bigram count matrix

        # Count bigrams
        for (i, j) in zip(self.char_codes_array[:-1], self.char_codes_array[1:]):
            bigram_counts[i, j] += 1

        # Normalize to create a probability distribution
        row_sums = bigram_counts.sum(axis=1, keepdims=True)  # Sum of each row
        self.probability_matrix = bigram_counts / row_sums  # Normalize
        self.probability_matrix = np.nan_to_num(self.probability_matrix)  # Handle NaN values

    def sample_next_char(self, current_char_index):
        """Sample the next character index based on the current character index."""
        probabilities = self.probability_matrix[current_char_index]
        next_char_index = np.random.choice(range(len(self.characters)), p=probabilities)
        return next_char_index

# Example usage
if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_bigram_matrix()
    current_index = 19  # Example starting character index
    next_index = model.sample_next_char(current_index)
    print("Current character index:", current_index)
    print("Current character:", model.characters[current_index])
    print("Next character index sampled:", next_index)
    print("Next character:", model.characters[next_index])
