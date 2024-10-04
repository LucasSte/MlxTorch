import torch
import numpy as np

torch.set_num_threads(1)

dev = torch.device("mps")
# np_arr = np.array([1, 2, 3, 4])
np_arr = [[1, 2], [3, 4]]
arr = torch.tensor(np_arr, dtype=torch.float32)
arr2 = arr.to(dev)
# arr9 = arr.to('cpu')


# print(arr9)
cpu_arr2 = [[5, 6], [7, 8]]
arr3 = torch.tensor(cpu_arr2, dtype=torch.float32)
arr6 = arr3.to(dev)
arr4 = torch.matmul(arr2, arr6)
#
# # arr5 = arr4.to(torch.device("cpu"))
print(arr4)
