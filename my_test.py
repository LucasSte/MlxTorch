import torch
import numpy as np

# dev = torch.device("mlx")
# np_arr = np.array([1, 2, 3, 4])
np_arr = [1, 2, 3, 4]
arr = torch.tensor(np_arr)
# arr2 = arr.to(dev)
print(arr)
