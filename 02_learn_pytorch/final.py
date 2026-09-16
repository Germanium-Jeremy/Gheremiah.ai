import torch
import torch.nn.functional as F

# ============================================================================
# STAGE 02 (final) — the bridge from counting to LEARNING.
# ----------------------------------------------------------------------------
# In 01/trigram.py the model IS the table (counts). Here a random weight
# matrix W plays the role of the table, and the pipeline
# one-hot -> matmul -> logits -> softmax -> cross-entropy -> .backward()
# is exactly the training recipe the whole workspace builds on.
# KEY INSIGHT: W @ one_hot(x) picks out ROW x of W — so a one-hot input times
# a matrix is just a (trainable) row lookup. Train that row and you've
# reimplemented the bigram table with gradient descent. This is the mental
# model behind nn.Embedding too.
# ============================================================================

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path
        self.more_path = more_path
        self.characters = []
        self.char_to_index = {}
        self.char_codes_tensor = None   # corpus as torch.long ids
        self.probability_matrix = None  # counted P(next|current) — baseline sampler
        self.W = None  # Weight matrix for the bigram model  # trainable (V, V): row x = logits for "next char" given current char x

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
            self.char_codes_tensor = torch.tensor(char_codes, dtype=torch.long)  # Convert to PyTorch tensor
        except FileNotFoundError:
            print("Dataset file not found.")

    def create_bigram_matrix(self):
        """Create a bigram count matrix and normalize it."""
        vocab_size = len(self.characters)  # Number of unique characters
        bigram_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)  # Bigram count matrix

        # Count bigrams
        # Shifted-pair zip as in 01: [:-1] with [1:] = every (current, next) pair.
        for (i, j) in zip(self.char_codes_tensor[:-1], self.char_codes_tensor[1:]):
            bigram_counts[i, j] += 1

        # Normalize to create a probability distribution
        # Row-normalize over dim=1 so row i = P(next | current=i).
        row_sums = bigram_counts.sum(dim=1, keepdim=True)  # Sum of each row
        self.probability_matrix = bigram_counts.float() / row_sums.float()  # Normalize
        self.probability_matrix[torch.isnan(self.probability_matrix)] = 0  # Handle NaN values

    def sample_next_char(self, current_char_index):
        """Sample the next character index based on the current character index."""
        probabilities = self.probability_matrix[current_char_index]
        next_char_index = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_index

    def train_linear_model(self, target_index):
        """Train a simple linear model using a random weight matrix."""
        vocab_size = len(self.characters)

        # Step 1: create a random weight matrix W of shape (vocab_size, vocab_size)
        # requires_grad=True = "this is a learnable parameter". From here on
        # torch records every op touching W so gradients can be computed.
        # REVISION NOTE: randn init is unnormalized; real layers use scaled
        # init (std=0.02, see stages 05/06) so early logits don't explode.
        self.W = torch.randn((vocab_size, vocab_size), requires_grad=True)

        # Step 2: create a one-hot vector for the current character index
        # one-hot = the "input encoding". Position target_index is 1, rest 0.
        # (In stage 04 this becomes F.one_hot over a whole batch.)
        one_hot_vector = torch.zeros(vocab_size, dtype=torch.float32)
        one_hot_vector[target_index] = 1.0

        # Step 3: perform matrix multiplication with W
        # W @ one_hot selects ROW target_index of W -> (V,) "logits":
        # raw, unnormalized scores, one per candidate next character.
        logits = self.W @ one_hot_vector  # Shape: (vocab_size,)

        # Step 4: apply softmax to get probabilities
        # softmax exponentiates + normalizes logits into a distribution that
        # sums to 1 — the model's guess for the next character.
        probabilities = F.softmax(logits, dim=0)
        print("Predicted probabilities for next character:\n", probabilities)

        # Step 5: compute cross entropy loss
        # cross_entropy measures -log(predicted prob of the correct next char):
        # 0 = perfect, ln(V) ≈ 4.17 = uniform guessing. It EXPECTS raw logits
        # (it applies log_softmax internally), which is why `logits` — not the
        # softmaxed `probabilities` — is passed in. Feeding softmaxed probs
        # instead would double-normalize and squash gradients.
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index]))
        print("Cross entropy loss:", loss.item())

        # Step 6: perform backpropagation
        # .backward() computes d(loss)/dW and accumulates it into W.grad.
        # (For comparison: real training would follow with an update step —
        # W -= lr * grad, or optimizer.step() as in stage 03 — and zero the
        # grads each iteration. This demo stops at the gradient on purpose.)
        loss.backward()
        print("Gradient of W:\n", self.W.grad)

if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_bigram_matrix()
    # 19 = the letter 'u' in this sorted vocab — same demo char as 01/final.py.
    current_index = 19  # Example starting character index
    model.train_linear_model(current_index)
    next_index = model.sample_next_char(current_index)
    print("Current character index:", current_index)
    print("Current character:", model.characters[current_index])
    print("Next character index sampled:", next_index)
    print("Next character:", model.characters[next_index])
    