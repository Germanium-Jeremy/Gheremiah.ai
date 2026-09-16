import torch, numpy as np

# ============================================================================
# STAGE 02 — Learn PyTorch: tensors + autograd.
# ----------------------------------------------------------------------------
# NumPy arrays (stage 01) can't remember how they were computed, so they can't
# be trained. PyTorch tensors CAN: with requires_grad=True every operation is
# recorded on a computation graph, and .backward() computes d(result)/d(input)
# automatically. That single feature is what makes neural-network training
# possible, and it's the whole point of this file.
# The tiny function used for the demo, y = x^2 + 2x + 1, has derivative
# dy/dx = 2x + 2 — easy to verify by hand against tensor.grad.
# ============================================================================


# Tensor creation and gradient tracking

# tensor: 2x2 float tensor created directly from a Python list.
# requires_grad=True is the switch that makes torch record every operation
# on this tensor so gradients can flow back to it. dtype=torch.float32 is the
# standard dtype for training (int tensors can't carry gradients).
tensor = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float32) # Creating a 2D tensor from a list of lists with gradient tracking
print("Original tensor:\n", tensor)

# Bridge from stage 01: torch.from_numpy shares MEMORY with the NumPy array
# (zero-copy) — mutating one mutates the other. Note: it does NOT set
# requires_grad; also, the shared array must stay alive while the tensor
# exists. Use torch.tensor(arr) instead if you want an independent copy.
arr = np.array([[1, 2], [3, 4]])
arr_tensor = torch.from_numpy(arr) # Creating a tensor from a NumPy array
print("Tensor from NumPy array:\n", arr_tensor)

# tensor_ones / tensor_zeros: pre-allocated tensors (like np.zeros/ones).
# requires_grad on constants like these is only useful if you intend to
# LEARN them — e.g. a bias vector or an embedding table is initialized
# exactly like this and updated by the optimizer during training.
tensor_ones = torch.ones((2, 3), requires_grad=True, dtype=torch.float32) # Creating a tensor filled with ones with shape (2, 3)
print("Tensor of ones:\n", tensor_ones)

tensor_zeros = torch.zeros((3, 2), requires_grad=True, dtype=torch.float32) # Creating a tensor filled with zeros with shape (3, 2)
print("Tensor of zeros:\n", tensor_zeros)

# Random initialization — this is how model weights start out. Uniform in
# [0, 1) here; real layers use smarter init (e.g. normal with std=0.02, see
# _init_weights in stages 05/06) because scale affects early training a lot.
tensor_random = torch.rand((2, 2), requires_grad=True, dtype=torch.float32) # Creating a tensor with random values between 0 and 1 with shape (2, 2)
print("Tensor with random values:\n", tensor_random)


# Performong operations on tensors

# Element-wise polynomial: y = x^2 + 2x + 1, i.e. (x + 1)^2 computed
# elementwise. Every op is recorded in the autograd graph built on `tensor`.
tensor_result = tensor ** 2 + 2 * tensor + 1 # Performing element-wise operations on the tensor. y = x^2 + 2x + 1

# .backward() walks that graph in reverse and fills tensor.grad.
# The argument is the "gradient of the output w.r.t. itself"; it's needed
# because tensor_result is non-scalar — torch.ones_like(...) means
# "d(total_sum)/d(each element)", i.e. dL/dy = 1 per element.
# REVISION NOTE: with a scalar loss you'd just call loss.backward().
# For y = x^2 + 2x + 1 the analytic gradient is 2x + 2, so:
#   x=1 -> 4, x=2 -> 6, x=3 -> 8, x=4 -> 10
# (verified against the printed gradient — this is the sanity check that
# autograd is doing what the calculus says.)
tensor_result.backward(torch.ones_like(tensor_result)) # Computing the gradients of the result with respect to the original tensor
print("Result of tensor operations:\n", tensor_result)
print("Gradient of the original tensor:\n", tensor.grad) # Printing the gradient of the original tensor
