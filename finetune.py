import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CNN,transform_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
dataset = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\train",
    transform=transform_1
    
)


cnn = CNN().to(device)

training_loader = DataLoader(
    dataset=dataset,
    batch_size=32, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
def finetune():
    cnn.load_state_dict(torch.load("model.pth"))
    optimizer = Adam(cnn.parameters(),lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,4.0]).to(device))
    num_epochs = 5
    for epoch in range(num_epochs):
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = cnn(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(cnn.state_dict(), "modelv2.pth")

if __name__ == "__main__":
    finetune()