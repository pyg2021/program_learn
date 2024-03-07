import torch
import ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context
# Data preparation
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    #.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对所有通道进行归一化，使其分布在[-1, 1]范围内
])

# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#
# #task1_data = [data for data in train_dataset if data[1] < 5]
# #task2_data = [data for data in train_dataset if data[1] >= 5]
# # Split data into two groups
# train_dataset_size = len(train_dataset)
# train_split_sizes = [train_dataset_size // 2, train_dataset_size - train_dataset_size // 2]
# task1_data, task2_data = random_split(train_dataset, train_split_sizes)
#
#
#
# task1_loader = DataLoader(task1_data, batch_size=64, shuffle=True)
# task2_loader = DataLoader(task2_data, batch_size=64, shuffle=True)
#
# test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
#
# #task1_test_data = [data for data in test_dataset if data[1] < 5]
# #task2_test_data = [data for data in test_dataset if data[1] >= 5]
# test_dataset_size = len(test_dataset)
# test_split_sizes = [test_dataset_size // 2, test_dataset_size - test_dataset_size // 2]
# task1_test_data, task2_test_data = random_split(test_dataset, test_split_sizes)
#
# task1_test_loader = DataLoader(task1_test_data, batch_size=64, shuffle=False)
# task2_test_loader = DataLoader(task2_test_data, batch_size=64, shuffle=False)


# 加载 MNIST 数据集
task1_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
task1_test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

task1_loader = DataLoader(task1_data , batch_size=64, shuffle=True)
task1_test_loader = DataLoader(task1_test_data, batch_size=64, shuffle=False)

# 加载 USPS 数据集
task2_data = datasets.USPS('./data', train=True, download=True, transform=transform)
task2_test_data = datasets.USPS('./data', train=False, download=True, transform=transform)
task2_loader = DataLoader(task2_data, batch_size=64, shuffle=True)
task2_test_loader = DataLoader(task2_test_data, batch_size=64, shuffle=False)


# Model definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# EWC implementation
class EWC:
    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
#计算fisher信息矩阵
    def _compute_fisher(self, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        self.model.train()
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = F.log_softmax(self.model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += (p.grad ** 2) / len(dataloader)

        return fisher

    def penalty(self, new_model):
        loss = 0
        for n, p in new_model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss * (self.importance / 2)


# Train function
def train(model, dataloader, optimizer, criterion, device, ewc=None, ewc_lambda=0.5):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_lambda * ewc_loss
            print(loss,ewc_lambda * ewc_loss)
        loss.backward()
        optimizer.step()
    


# Test function
def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = SimpleNet().to(device)

# Train on Task 1
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
for epoch in range(2):
    train(model, task1_loader, optimizer, criterion, device)
task1_accuracy = test(model, task1_test_loader, device)
print(f'Task 1 accuracy: {task1_accuracy}%')
task2_accuracy_NEW = test(model, task2_test_loader, device)
print(f'Tasknew 2 accuracy: {task2_accuracy_NEW}%')
# Save EWC
ewc = EWC(model, task1_loader, device)

# Train on Task 2
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(10):
    train(model, task2_loader, optimizer, criterion, device, ewc=ewc, ewc_lambda=10 )
    task1_accuracy_new = test(model, task1_test_loader, device)
    print(f'Tasknew 1 accuracy: {task1_accuracy_new}%')
task2_accuracy = test(model, task2_test_loader, device)

print(f'Task 2 accuracy: {task2_accuracy}%')

# Train on Task 2 but don't have ewc
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# for epoch in range(10):
#     #训练ewc=none代表不使用ewc算法
#     train(model, task2_loader, optimizer, criterion, device, ewc=None)
# task2_accuracy = test(model, task2_test_loader, device)
#
# print(f'Task 2 dont have ewc accuracy: {task2_accuracy}%')

task1_accuracy_new = test(model, task1_test_loader, device)
print(f'Tasknew 1 accuracy: {task1_accuracy_new}%')
task2_accuracy_NEW = test(model, task2_test_loader, device)
print(f'Tasknew 2 accuracy: {task2_accuracy_NEW}%')