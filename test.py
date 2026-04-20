import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 
from torchvision import transforms
from model import CNN
from sklearn.metrics import classification_report,recall_score,f1_score,precision_score,confusion_matrix
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ]
)
val_dataset= ImageFolder(
    root=r"E:\datasets\archive\chest_xray\test",
    transform=transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True

)
def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load("modelv3.pth"))
    cnn.eval()
    model_predicts=[]
    true_predicts=[]
    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = cnn(images)
            _,predicted = torch.max(output,1)
            model_predicts.extend(predicted.cpu().numpy())
            true_predicts.extend(labels.cpu().numpy())
    print(classification_report(model_predicts,true_predicts))
    print("Precision:",precision_score(model_predicts,true_predicts))
    print("Recall:",recall_score(model_predicts,true_predicts))
    print("F1 Score:",f1_score(model_predicts,true_predicts))
    print("Confusion Matrix:")
    print(confusion_matrix(model_predicts,true_predicts))


if __name__ == "__main__":
    validate()

