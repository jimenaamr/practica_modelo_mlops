import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow

# ----- 0) Config MLflow -----
# (opcional) mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mnist-simple-nn")

# ----- 1) Datos -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True,  transform=transform, download=False)
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=False)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ----- 2) Modelo -----
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# ----- 3) Pérdida y optimizador -----
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ----- 4) Entrenamiento + MLflow -----
epochs = 5
with mlflow.start_run():
    # log de hiperparámetros
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model", "SimpleNN(784-128-10)")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

    # ----- 5) Evaluación -----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    mlflow.log_metric("test_accuracy", accuracy)

    # Guardar el modelo en MLflow
    mlflow.pytorch.log_model(model, artifact_path="model")
