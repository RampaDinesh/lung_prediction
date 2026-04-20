from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ]
)

transform_1 = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5,std=0.5)
    
])




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*26*26,out_features=128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=2)
        )
    def forward(self,x):
        x=self.cnn_layer(x)
        x=self.fc(x)
        return x










def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\train",
    transform=transform_1
    
)
    dataset_test = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\test",
    transform=transform_1
    
)
    training_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
   
)
    test_loader = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False
)
    c1 = CNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0]).to(device))
    optimizer = torch.optim.Adam(c1.parameters(),lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = c1(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(c1.state_dict(), "modelv3.pth")
    

if __name__ == "__main__":
    main()
