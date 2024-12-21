import torch
import time

start = time.time()

dev = torch.device("mlx")
cpu_arr = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
arr = torch.tensor(cpu_arr, dtype=torch.float32)
arr2 = arr.to(dev)

cpu_arr2 = [[26, 27, 28, 29, 30], [31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40], [41, 42, 43, 44, 45], [46, 47, 48, 49, 50]]
arr3 = torch.tensor(cpu_arr2, dtype=torch.float32)
arr6 = arr3.to(dev)
arr4 = torch.matmul(arr2, arr6)

end = time.time()

print(arr4)
print(f'Time: {end-start}s')
