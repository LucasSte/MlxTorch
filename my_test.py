import torch
import numpy as np

dev = torch.device("mlx")
# np_arr = np.array([1, 2, 3, 4])
np_arr = [[1, 2], [3, 4]]
arr = torch.tensor(np_arr, device=dev, dtype=torch.float32)
# arr2 = arr.to(dev)
# arr3 = arr.to(torch.device('cpu'))

cpu_arr2 = [[5, 6], [7, 8]]
arr3 = torch.tensor(cpu_arr2, device=dev, dtype=torch.float32)
arr4 = torch.matmul(arr, arr3)
print(arr4)
