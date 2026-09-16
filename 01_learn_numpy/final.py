import numpy as np

# ============================================================================
# STAGE 01 (final) — the bigram model refactored into a class.
# ----------------------------------------------------------------------------
# Same math as main.py, but packaged as TextModel so state (vocab, encodings,
# probability matrix) lives on the instance instead of loose globals.
# This class shape is the template: every later stage (02, 03) keeps this
# class but swaps the counting for tensor math and gradient learning.
# ============================================================================

class TextModel:
    # __init__ only sets up empty state; real work happens in the methods.
    # Instance attributes declared up front so readers (and linters) see the
    # object's shape before any method runs.
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path  # path to corpus 1
        self.more_path = more_path                          # path to corpus 2
        self.characters = []          # vocab list (index -> char); len = vocab_size
        self.char_to_index = {}       # tokenizer dict (char -> index)
        self.char_codes_array = None  # the whole corpus as integer ids (np array)
        self.probability_matrix = None  # P(next | current), shape (vocab, vocab)

    def load_data(self):
        """Load the datasets and create character codes."""
        try:
            with open(self.tiny_shakespeare_path, "r") as f:
                lines = f.read()

            # Concatenating both corpora enlarges the training data; vocab is
            # built AFTER merging so it covers characters from either file.
            with open(self.more_path, "r") as f:
                lines += f.read()

            # sorted(set(...)) = deterministic vocab: same corpus -> same
            # index for each char on every run. Keep this stable if you
            # compare outputs across runs or load saved models later.
            self.characters = sorted(set(lines))  # Get unique characters
            self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}  # Mapping from char to index
            char_codes = [self.char_to_index[char] for char in lines]  # Convert to indices
            self.char_codes_array = np.array(char_codes)  # Convert to NumPy array
            # REVISION NOTE: any character outside this vocab would KeyError
            # here — the model can never emit a char it has never seen.
        except FileNotFoundError:
            print("Dataset file not found.")

    def create_bigram_matrix(self):
        """Create a bigram count matrix and normalize it."""
        # Local vocab_size (len of vocab) — sets both dimensions of the model.
        vocab_size = len(self.characters)  # Number of unique characters
        # Counts: bigram_counts[i, j] = how often char i is followed by char j.
        bigram_counts = np.zeros((vocab_size, vocab_size), dtype=int)  # Bigram count matrix

        # Count bigrams
        # The shifted-pair trick: [:-1] paired with [1:] gives every
        # (current, next) couple in the corpus — this is the whole dataset.
        for (i, j) in zip(self.char_codes_array[:-1], self.char_codes_array[1:]):
            bigram_counts[i, j] += 1

        # Normalize to create a probability distribution
        # Each row of counts -> conditional distribution P(next | current).
        # keepdims=True + broadcasting = divide each row by its own sum.
        row_sums = bigram_counts.sum(axis=1, keepdims=True)  # Sum of each row
        self.probability_matrix = bigram_counts / row_sums  # Normalize
        # REVISION NOTE: integer 0 rows (char never appears as "current")
        # would produce NaN; NaN breaks np.random.choice(p=...), so map to 0.
        self.probability_matrix = np.nan_to_num(self.probability_matrix)  # Handle NaN values
        # Big-picture: this matrix IS the learned model. The neural stages
        # (02 final.py, 03) try to approximate the same P(next | current)
        # with trainable weights instead of direct counting.

    def sample_next_char(self, current_char_index):
        """Sample the next character index based on the current character index."""
        # Row lookup = the distribution over the next char given the current one.
        probabilities = self.probability_matrix[current_char_index]
        # torch.multinomial / np.random.choice equivalent — random draw
        # proportional to probability. Sampling (not argmax) keeps generation
        # from becoming a deterministic repeating loop.
        # ALTERNATIVE (as a comment, for reproducible runs):
        #   rng = np.random.default_rng(seed)
        #   next_char_index = rng.choice(len(self.characters), p=probabilities)
        # np.random.choice uses global state; a seeded default_rng makes runs
        # reproducible — handy when comparing model variants.
        next_char_index = np.random.choice(range(len(self.characters)), p=probabilities)
        return next_char_index

# Example usage
# Standard guard: run the demo only when executed directly, not on import.
if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_bigram_matrix()
    # 19 = the letter 'u' in this sorted vocab (spot in main.py's demo).
    current_index = 19  # Example starting character index
    next_index = model.sample_next_char(current_index)
    print("Current character index:", current_index)
    print("Current character:", model.characters[current_index])
    print("Next character index sampled:", next_index)
    print("Next character:", model.characters[next_index])
