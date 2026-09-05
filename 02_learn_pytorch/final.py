import torch
import torch.nn.functional as F

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path
        self.more_path = more_path
        self.characters = []
        self.char_to_index = {}
        self.char_codes_tensor = None
        self.probability_matrix = None
        self.W = None  # Weight matrix for the bigram model

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
        for (i, j) in zip(self.char_codes_tensor[:-1], self.char_codes_tensor[1:]):
            bigram_counts[i, j] += 1

        # Normalize to create a probability distribution
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
        self.W = torch.randn((vocab_size, vocab_size), requires_grad=True)

        # Step 2: create a one-hot vector for the current character index
        one_hot_vector = torch.zeros(vocab_size, dtype=torch.float32)
        one_hot_vector[target_index] = 1.0

        # Step 3: perform matrix multiplication with W
        logits = self.W @ one_hot_vector  # Shape: (vocab_size,)

        # Step 4: apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=0)
        print("Predicted probabilities for next character:\n", probabilities)

        # Step 5: compute cross entropy loss
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index]))
        print("Cross entropy loss:", loss.item())

        # Step 6: perform backpropagation
        loss.backward()
        print("Gradient of W:\n", self.W.grad)

if __name__ == "__main__":
    model = TextModel("dataset/input.txt", "dataset/more.txt")
    model.load_data()
    model.create_bigram_matrix()
    current_index = 19  # Example starting character index
    model.train_linear_model(current_index)
    next_index = model.sample_next_char(current_index)
    print("Current character index:", current_index)
    print("Current character:", model.characters[current_index])
    print("Next character index sampled:", next_index)
    print("Next character:", model.characters[next_index])
    