from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import torch
from torchvision.models import resnet18,ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(in_features=model.fc.in_features,out_features=2)
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\train",
    transform=transform_train
)

trainer_loader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=32,
    pin_memory=True,
    num_workers=4


)

model.to(device)
optimizer = Adam(params=model.parameters(),lr=0.0001)
entropy = nn.CrossEntropyLoss(weight=torch.tensor([2.0,1.0]).to(device=device))


def main():
    model.train()
    num_epoches=10
    for i in range(num_epoches):
        for image,label in trainer_loader:
            image=image.to(device)
            label= label.to(device)
            output = model(image)
            loss = entropy(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoches[{i+1}/{num_epoches}] loss: {loss.item():.4f}")
    torch.save(model.state_dict(),"resent182.pth")


if __name__ == "__main__":
    main()