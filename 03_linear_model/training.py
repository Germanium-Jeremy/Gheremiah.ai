import torch
import torch.nn.functional as F
import os
import math

class TextModel:
    def __init__(self, tiny_shakespeare_path, more_path):
        self.tiny_shakespeare_path = tiny_shakespeare_path
        self.more_path = more_path
        self.characters = []
        self.char_to_index = {}
        self.char_codes_tensor = None
        self.probability_matrix = None
        self.W = None  # Weight matrix

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

    def generate_text(self, start_index, length=200):
        """Generate text of a given length starting from a specific character index."""
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
        self.W = torch.randn(vocab_size, vocab_size, requires_grad=True)  # Initialize weight matrix
        optimizer = torch.optim.SGD([self.W], lr=learning_rate)  # Simple SGD optimizer

        total_loss = 0.0
        num_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(len(self.char_codes_tensor) - 1):
                current_index = self.char_codes_tensor[i]
                target_index = self.char_codes_tensor[i + 1]

                # Create a one-hot vector for the current character index
                one_hot_vector = torch.zeros(vocab_size, dtype=torch.float32)
                one_hot_vector[current_index] = 1.0  # Set the target index to 1

                # Perform matrix multiplication with W
                logits = self.W @ one_hot_vector  # Shape will be (vocab_size,)

                # Apply softmax to get predicted probabilities
                predicted_probs = F.softmax(logits, dim=0)

                # Compute cross-entropy loss
                loss = F.cross_entropy(predicted_probs.unsqueeze(0), torch.tensor([target_index]))  # unsqueeze for batch size
                epoch_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                num_samples += 1

            average_loss = epoch_loss / num_samples
            perplexity = math.exp(average_loss)  # Calculate perplexity
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}")

    def save_model(self, model_name):
        """Save the model to a file."""
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
    model.train_linear_model(epochs=10)

    # Save the model
    model.save_model("01_gheremiah_ai_learned_pytorch")

    # Load the model
    model.load_model("01_gheremiah_ai_learned_pytorch")
    
    # Generate text with autoregressive method
    generated_text = model.autoregressive_generate_text("A", length=300)  # Start with a seed character
    print("Generated text:", generated_text)
