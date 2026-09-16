import torch
import torch.nn.functional as F

# ============================================================================
# STAGE 02 — TRIGRAM: same counting idea as 01, but with TWO chars of context.
# ----------------------------------------------------------------------------
# Bigram (01): P(next | current)        -> matrix (V, V)
# Trigram (here): P(next | prev, current) -> tensor (V, V, V)
# One more context token = one more matrix dimension. That's exactly why real
# models use neural networks instead: N-gram tables grow exponentially
# (V^N) and go sparse, while networks share parameters across contexts.
# Also the first taste of PyTorch tensor ops standing in for NumPy ones.
# (Note: `self.W` is declared but unused — this version is pure counting;
# the learned-weight version arrives in final.py / stage 03.)
# ============================================================================

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path  # corpus path 1
        self.more_path = more_path                          # corpus path 2
        self.characters = []            # vocab (index -> char)
        self.char_to_index = {}         # tokenizer dict (char -> index)
        self.char_codes_tensor = None   # corpus as a torch.long tensor of ids
        self.probability_matrix = None  # P(next | prev, current): (V, V, V)
        self.W = None  # Weight matrix for the trigram model  # (unused in this counting version — kept for API parity with final.py)

    def load_data(self):
        """Load the datasets and create character codes."""
        try:
            with open(self.tiny_shakespeare_path, "r") as f:
                lines = f.read()

            with open(self.more_path, "r") as f:
                lines += f.read()

            # Deterministic vocab: sorted(set(text)). Its length V sizes the
            # trigram tensor (V^3 entries!) and every later weight matrix.
            self.characters = sorted(set(lines))  # Get unique characters
            self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}  # Mapping from char to index
            char_codes = [self.char_to_index[char] for char in lines]  # Convert to indices
            # dtype=torch.long because these are INDICES (for tensor indexing),
            # not values to do math on.
            self.char_codes_tensor = torch.tensor(char_codes, dtype=torch.long)  # Convert to PyTorch tensor
        except FileNotFoundError:
            print("Dataset file not found.")

    def create_trigram_matrix(self):
        """Create a trigram count matrix and normalize it."""
        # "Matrix" here means the (V, V, V) count TENSOR — one extra dimension
        # for the extra context character vs the bigram version in 01.
        vocab_size = len(self.characters)  # Number of unique characters
        # (V, V, V) counts: entry [i, j, k] = times (i, j) was followed by k.
        # For V=65 that's 274,625 cells — most stay 0 (sparsity problem that
        # neural nets solve by parameter sharing).
        trigram_counts = torch.zeros((vocab_size, vocab_size, vocab_size), dtype=torch.int32)  # Trigram count matrix

        # Count trigrams
        # THREE shifted slices zipped together: [:-2], [1:-1], [2:] give every
        # (prev, current, next) triple — the trigram dataset in one line.
        # (Bigram version in 01 zipped just two slices.)
        for (i, j, k) in zip(self.char_codes_tensor[:-2], self.char_codes_tensor[1:-1], self.char_codes_tensor[2:]):
            trigram_counts[i, j, k] += 1

        # Normalize to create a probability distribution
        # Normalize along the LAST dim (dim=2): for fixed (i, j) the slice
        # [i, j, :] becomes P(next | prev=i, current=j).
        row_sums = trigram_counts.sum(dim=2, keepdim=True)  # Sum of each row for the last dimension
        self.probability_matrix = trigram_counts.float() / row_sums.float()  # Normalize
        # REVISION NOTE: integer division-by-zero -> NaN would break
        # torch.multinomial; replace unseen (i, j) slices with 0 probability.
        # ALTERNATIVE: row_sums.clamp(min=1e-8) before dividing (the trick
        # used in 03/training.ipynb) — no separate NaN pass needed.
        self.probability_matrix[torch.isnan(self.probability_matrix)] = 0  # Handle NaN values

    def sample_next_char(self, prev_char_index, current_char_index):
        """Sample the next character index based on the previous two character indices."""
        # Index with the (prev, current) PAIR -> a (V,) distribution over the
        # next char. torch.multinomial draws one index with probability
        # proportional to the distribution (NumPy's np.random.choice twin).
        probabilities = self.probability_matrix[prev_char_index, current_char_index]
        next_char_index = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_index

    def predict_next_characters(self, start_string, length=100):
        """Predict the next characters based on the starting string."""
        generated = start_string
        for _ in range(length):
            # Get indices for the last two characters
            # The sliding context: the LAST TWO chars of the generated string
            # are always the conditioning pair (this is what makes the model
            # "trigram"). Needs >= 2 chars of history to even start.
            if len(generated) < 2:
                # If the generated string is less than 2 characters, we can't predict
                break
            prev_char_index = self.char_to_index[generated[-2]]
            current_char_index = self.char_to_index[generated[-1]]
            
            # Sample the next character
            next_index = self.sample_next_char(prev_char_index, current_char_index)
            generated += self.characters[next_index]  # Append the predicted character
        
        return generated

if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_trigram_matrix()
    
    # Example usage of predicting the next characters
    # Autoregressive loop: append prediction, it becomes part of the next
    # context. Same generate() pattern reused in every later stage.
    start_string = "he"  # Example starting string
    generated_text = model.predict_next_characters(start_string, length=100)
    
    print("Generated text:\n", generated_text)
