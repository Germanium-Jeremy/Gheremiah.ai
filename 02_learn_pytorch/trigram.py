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
        self.W = None  # Weight matrix for the trigram model

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

    def create_trigram_matrix(self):
        """Create a trigram count matrix and normalize it."""
        vocab_size = len(self.characters)  # Number of unique characters
        trigram_counts = torch.zeros((vocab_size, vocab_size, vocab_size), dtype=torch.int32)  # Trigram count matrix

        # Count trigrams
        for (i, j, k) in zip(self.char_codes_tensor[:-2], self.char_codes_tensor[1:-1], self.char_codes_tensor[2:]):
            trigram_counts[i, j, k] += 1

        # Normalize to create a probability distribution
        row_sums = trigram_counts.sum(dim=2, keepdim=True)  # Sum of each row for the last dimension
        self.probability_matrix = trigram_counts.float() / row_sums.float()  # Normalize
        self.probability_matrix[torch.isnan(self.probability_matrix)] = 0  # Handle NaN values

    def sample_next_char(self, prev_char_index, current_char_index):
        """Sample the next character index based on the previous two character indices."""
        probabilities = self.probability_matrix[prev_char_index, current_char_index]
        next_char_index = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_index

    def predict_next_characters(self, start_string, length=100):
        """Predict the next characters based on the starting string."""
        generated = start_string
        for _ in range(length):
            # Get indices for the last two characters
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
    start_string = "he"  # Example starting string
    generated_text = model.predict_next_characters(start_string, length=100)
    
    print("Generated text:\n", generated_text)
