import torch, numpy as np


# Tensor creation and gradient tracking
tensor = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float32) # Creating a 2D tensor from a list of lists with gradient tracking
print("Original tensor:\n", tensor)

arr = np.array([[1, 2], [3, 4]])
arr_tensor = torch.from_numpy(arr) # Creating a tensor from a NumPy array
print("Tensor from NumPy array:\n", arr_tensor)

tensor_ones = torch.ones((2, 3), requires_grad=True, dtype=torch.float32) # Creating a tensor filled with ones with shape (2, 3)
print("Tensor of ones:\n", tensor_ones)

tensor_zeros = torch.zeros((3, 2), requires_grad=True, dtype=torch.float32) # Creating a tensor filled with zeros with shape (3, 2)
print("Tensor of zeros:\n", tensor_zeros)

tensor_random = torch.rand((2, 2), requires_grad=True, dtype=torch.float32) # Creating a tensor with random values between 0 and 1 with shape (2, 2)
print("Tensor with random values:\n", tensor_random)


# Performong operations on tensors
tensor_result = tensor ** 2 + 2 * tensor + 1 # Performing element-wise operations on the tensor. y = x^2 + 2x + 1
tensor_result.backward(torch.ones_like(tensor_result)) # Computing the gradients of the result with respect to the original tensor
print("Result of tensor operations:\n", tensor_result)
print("Gradient of the original tensor:\n", tensor.grad) # Printing the gradient of the original tensor

