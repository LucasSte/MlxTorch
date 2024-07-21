import torch


dev = torch.device("mlx")
arr = torch.tensor([1, 2, 3, 4], device=dev)
