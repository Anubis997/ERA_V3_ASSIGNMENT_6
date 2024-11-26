import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

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
            
            nn.Conv2d(12, 24, kernel_size=3, padding=1),  # 14x14x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7x24
            
            nn.Conv2d(24, 36, kernel_size=3, padding=1),  # 7x7x36
            nn.BatchNorm2d(36),
            nn.ReLU(),
            
            nn.Conv2d(36, 24, kernel_size=3, padding=1),  # 7x7x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            nn.Dropout2d(0.15)
        )
        
        # Replace fully connected layers with a GAP layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),  # Flatten the output
            nn.Linear(24, 10)  # Output layer for 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data transformations for better accuracy with data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
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
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel has {param_count} parameters")
    
    if param_count >= 20000:
        raise ValueError(f"Model has {param_count} parameters, which exceeds the limit of 20,000")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Open a log file to write training logs
    with open("training_logs.txt", "w") as log_file:
        num_epochs = 20
        consecutive_epochs_above_threshold = 0
        accuracy_threshold = 99.4
        for epoch in range(num_epochs):
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            total_batches = len(trainloader)
            
            log_file.write(f'\nEpoch: {epoch + 1}/{num_epochs}\n')
            log_file.write('-' * 60 + '\n')
            
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
            average_loss = running_loss / total_batches
            
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
            
            # Log the results
            log_file.write(f'Loss: {average_loss:.4f} | Train Accuracy: {final_train_accuracy:.2f}% | Test Accuracy: {final_test_accuracy:.2f}%\n')
            
            if final_test_accuracy >= accuracy_threshold:
                consecutive_epochs_above_threshold += 1
                log_file.write(f'Test Accuracy above {accuracy_threshold}% for {consecutive_epochs_above_threshold} consecutive epochs.\n')
            else:
                consecutive_epochs_above_threshold = 0
            
            if consecutive_epochs_above_threshold > 2:
                log_file.write(f'Stopping training early after {epoch + 1} epochs due to high test accuracy.\n')
                break
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    return model, final_test_accuracy

def update_readme(accuracy):
    readme_path = "README.md"
    with open(readme_path, "r") as file:
        content = file.readlines()

    # Find the line with the accuracy placeholder and replace it
    for i, line in enumerate(content):
        if "The model achieved a test accuracy of" in line:
            content[i] = f"The model achieved a test accuracy of **{accuracy:.2f}%** on the validation dataset.\n"
            break

    # Write the updated content back to README.md
    with open(readme_path, "w") as file:
        file.writelines(content)

if __name__ == "__main__":
    model, accuracy = train_model()