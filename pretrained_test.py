import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, confusion_matrix


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = ImageFolder(
    root=r"E:\datasets\archive\chest_xray\test",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    
    model.load_state_dict(torch.load("resent182.pth", map_location=device))
    model = model.to(device)

    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            #_, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs,1)
            predicted= (probs[:,1]>0.9).long()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    
    print(classification_report(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    validate()