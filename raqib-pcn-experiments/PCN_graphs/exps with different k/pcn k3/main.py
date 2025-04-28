# import time

# for i in range(100000):
#     print(i)
#     time.sleep(60)


import torch

def get_device():
    """Returns the device to be used ('cuda' if GPU is available, otherwise 'cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
