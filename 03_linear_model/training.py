import torch
import torch.nn.functional as F
import os
import math

# ============================================================================
# STAGE 03 — Train a linear bigram model with gradient descent.
# ----------------------------------------------------------------------------
# Stage 02 showed one forward+backward pass. This file turns it into a full
# training loop: many examples, an optimizer, averaged loss, perplexity,
# and save/load. The model is still just a (V, V) matrix W — but now it is
# LEARNED from data instead of counted, which is the point to compare
# against the bigram count matrix in 01.
# NOTE: the per-sample loop is deliberately unoptimized (no batching, no
# matmul-over-dataset tricks) so every training step is visible. Batching
# arrives in stage 04.
# ============================================================================

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path  # corpus path 1
        self.more_path = more_path                          # corpus path 2
        self.characters = []            # vocab (index -> char); len = V
        self.char_to_index = {}         # tokenizer dict (char -> index)
        self.char_codes_tensor = None   # corpus as torch.long ids
        self.probability_matrix = None  # counted P(next|current) — kept as a baseline sampler
        self.W = None  # Weight matrix  # learned (V, V): row x = logits for next char given current char x

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
        # Shifted-pair zip (see 01): [:-1] with [1:] = every (current, next) pair.
        for (i, j) in zip(self.char_codes_tensor[:-1], self.char_codes_tensor[1:]):
            bigram_counts[i, j] += 1

        # Normalize to create a probability distribution
        # Row i becomes P(next | current=i); keepdims enables row-wise broadcasting.
        row_sums = bigram_counts.sum(dim=1, keepdim=True)  # Sum of each row
        self.probability_matrix = bigram_counts.float() / row_sums.float()  # Normalize
        # NaN (never-seen current char) -> 0 so multinomial sampling stays valid.
        self.probability_matrix[torch.isnan(self.probability_matrix)] = 0  # Handle NaN values

    def sample_next_char(self, current_char_index):
        """Sample the next character index based on the current character index."""
        # Samples from the COUNTED matrix. Interesting experiment (as a
        # comment): swap in the LEARNED W to sample from the trained model:
        #   probs = F.softmax(self.W[current_char_index], dim=0)
        # then compare fluency — that's the counting-vs-learning comparison.
        probabilities = self.probability_matrix[current_char_index]
        next_char_index = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_index

    def generate_text(self, start_index, length=200):
        """Generate text of a given length starting from a specific character index."""
        # Autoregressive loop: predict one char, append it, feed it back as
        # context. Each sample depends on all previous ones — hence
        # "autoregressive". (Keeps state via current_index rather than a
        # string like 02/trigram.py; same idea, tensor-friendly.)
        current_index = start_index
        generated_text = [self.characters[current_index]]

        for _ in range(length - 1):
            next_index = self.sample_next_char(current_index)
            generated_text.append(self.characters[next_index])
            current_index = next_index

        return ''.join(generated_text)

    def train_linear_model(self, epochs=100, learning_rate=0.01):
        """Train a simple linear model using a random weight matrix."""
        vocab_size = len(self.characters)
        # Random init of the learnable (V, V) matrix. requires_grad=True
        # registers W with autograd so loss.backward() fills W.grad.
        self.W = torch.randn(vocab_size, vocab_size, requires_grad=True)  # Initialize weight matrix
        # SGD: W -= lr * gradient, the plainest optimizer. lr is the step
        # size — too high diverges, too low crawls; 0.01 is a safe default.
        optimizer = torch.optim.SGD([self.W], lr=learning_rate)  # Simple SGD optimizer

        total_loss = 0.0
        num_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            # One epoch = a full pass over every bigram in the corpus.
            for i in range(len(self.char_codes_tensor) - 1):
                current_index = self.char_codes_tensor[i]
                target_index = self.char_codes_tensor[i + 1]
                # (current, target) is one training example: given the
                # current char, predict the next one.

                # Create a one-hot vector for the current character index
                # W @ one_hot(x) == row x of W (see 02/final.py) — this is
                # the entire "layer" of the model: a lookup of learned logits.
                one_hot_vector = torch.zeros(vocab_size, dtype=torch.float32)
                one_hot_vector[current_index] = 1.0  # Set the target index to 1

                # Perform matrix multiplication with W
                logits = self.W @ one_hot_vector  # Shape will be (vocab_size,)

                # Apply softmax to get predicted probabilities
                # NOTE (revision): predicted_probs is computed for inspection
                # only — cross_entropy below is fed the raw logits (it applies
                # log_softmax internally). Feeding the softmaxed probs here
                # instead would double-normalize and squash the gradients.
                # The cleaner equivalent: delete this line and pass logits
                # straight to cross_entropy (as this code already does).
                predicted_probs = F.softmax(logits, dim=0)

                # Compute cross-entropy loss
                # -log P(target | current): pushes up the logit of the true
                # next char, pushes down all others. unsqueeze(0) adds a
                # batch dimension because cross_entropy wants (batch, V) vs
                # (batch,) — batch of 1 here.
                loss = F.cross_entropy(predicted_probs.unsqueeze(0), torch.tensor([target_index]))  # unsqueeze for batch size
                epoch_loss += loss.item()

                # Backpropagation
                # The canonical three lines of PyTorch training:
                #   zero_grad — clear W.grad accumulated by the last step
                #   backward  — compute d(loss)/dW via autograd
                #   step      — SGD update: W -= lr * W.grad
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                num_samples += 1

            # REVISION NOTE: average_loss uses num_samples (the running TOTAL
            # across all epochs so far), not the count for just this epoch —
            # so later epochs' printed averages get pulled toward the global
            # mean. For a per-epoch figure the alternative is dividing by the
            # number of steps taken in THIS epoch only.
            average_loss = epoch_loss / num_samples
            # Perplexity = exp(mean cross-entropy): "how many equally likely
            # choices the model is torn between". Uniform guessing = V ≈ 65;
            # lower is better and each epoch should trend down.
            perplexity = math.exp(average_loss)  # Calculate perplexity
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}")

    def save_model(self, model_name):
        """Save the model to a file."""
        # Checkpointing: W plus everything needed to USE the model (vocab and
        # the count-based sampler). Saving the mapping matters — indices are
        # meaningless without the same char_to_index at load time.
        model_path = f"{model_name}.pt"
        torch.save({
            'W': self.W,
            'characters': self.characters,
            'char_to_index': self.char_to_index,
            'probability_matrix': self.probability_matrix
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_name):
        """Load the model from a file."""
        # Restores a checkpoint saved above. REVISION NOTE: torch.load
        # (without weights_only=True, as here) executes arbitrary pickled
        # code — fine for your own checkpoints, but prefer
        # torch.load(path, weights_only=True) for files from elsewhere.
        model_path = f"{model_name}.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.W = checkpoint['W']
            self.characters = checkpoint['characters']
            self.char_to_index = checkpoint['char_to_index']
            self.probability_matrix = checkpoint['probability_matrix']
            print(f"Model loaded from {model_path}")
        else:
            print("Model file does not exist.")

    def autoregressive_generate_text(self, seed, length=200):
        """Generate text using autoregressive generation."""
        # Same loop as generate_text, but takes a seed STRING and maps it
        # through the tokenizer dict — the friendlier entry point.
        # REVISION NOTE: a seed character absent from the vocab would raise
        # KeyError; unseen-character handling (e.g. an <UNK> token or a
        # try/except fallback) arrives with BPE tokenization in stage 04.
        current_index = self.char_to_index[seed]
        generated_text = [seed]

        for _ in range(length - 1):
            next_index = self.sample_next_char(current_index)
            generated_text.append(self.characters[next_index])
            current_index = next_index

        return ''.join(generated_text)

if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_bigram_matrix()

    # Train the linear model
    # 10 epochs is a smoke run; loss should fall visibly from ln(65) ≈ 4.17.
    model.train_linear_model(epochs=10)

    # Save the model
    model.save_model("01_gheremiah_ai_learned_pytorch")

    # Load the model
    model.load_model("01_gheremiah_ai_learned_pytorch")
    
    # Generate text with autoregressive method
    # 'A' is a natural seed — capital letters start sentences/speaker names.
    generated_text = model.autoregressive_generate_text("A", length=300)  # Start with a seed character
    print("Generated text:", generated_text)
