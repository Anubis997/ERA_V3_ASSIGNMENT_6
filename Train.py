import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 28x28x4
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 28x28x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14x8
            nn.Conv2d(8, 12, kernel_size=3, padding=1),  # 14x14x12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # 14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 7x7x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),  # Flatten the output
            nn.Linear(32, 10)  # Output layer for 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    
    train_size = 50000
    test_size = 10000
    trainset, testset = torch.utils.data.random_split(full_trainset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=4)
    
    model = MNIST_CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    num_epochs = 20
    consecutive_epochs_above_threshold = 0
    accuracy_threshold = 99.4
    training_logs = []  # To capture training logs

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        print(f'\nEpoch: {epoch + 1}/{num_epochs}')
        print('-' * 60)
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        final_train_accuracy = 100. * correct / total
        average_loss = running_loss / len(trainloader)
        
        # Calculate test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, test_predicted = output.max(1)
                test_total += target.size(0)
                test_correct += test_predicted.eq(target).sum().item()
        
        final_test_accuracy = 100. * test_correct / test_total
        
        # Capture training logs
        training_logs.append(f'Epoch: {epoch + 1}, Loss: {average_loss:.4f}, Train Accuracy: {final_train_accuracy:.2f}%, Test Accuracy: {final_test_accuracy:.2f}%')
        
        print(f'Loss: {average_loss:.4f} | Train Accuracy: {final_train_accuracy:.2f}% | Test Accuracy: {final_test_accuracy:.2f}%')
        
        if final_test_accuracy >= accuracy_threshold:
            consecutive_epochs_above_threshold += 1
            print(f'Test Accuracy above {accuracy_threshold}% for {consecutive_epochs_above_threshold} consecutive epochs.')
        else:
            consecutive_epochs_above_threshold = 0
        
        if consecutive_epochs_above_threshold > 2:
            print(f'Stopping training early after {epoch + 1} epochs due to high test accuracy.')
            break
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    return model, final_test_accuracy, training_logs

if __name__ == "__main__":
    model, accuracy, logs = train_model()
    for log in logs:
        print(log)  # Print training logs