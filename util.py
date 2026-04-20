from collections import Counter
from torchvision.datasets import ImageFolder
dataset = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\train",
    
    
)

print(dataset.class_to_idx)
print(Counter(dataset.targets))