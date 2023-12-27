import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 检查是否有可用的GPU，如果有的话使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 教师网络结构
class Teacher_model(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(Teacher_model, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 学生网络结构
class Student_model(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(Student_model, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 数据集加载和预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# MNIST 数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 数据集分割
train_length = int(len(mnist_dataset) * 0.5)
test_length = int(len(mnist_dataset) * 0.25)
val_length = len(mnist_dataset) - train_length - test_length
train_dataset, test_dataset, val_dataset = random_split(mnist_dataset, [train_length, test_length, val_length])

# 创建未标记的数据集，这里我们将一部分训练数据的标签置空
unlabeled_length = int(0.5 * train_length)  # 取一半的训练数据作为未标记数据
indices = torch.randperm(len(train_dataset))[:unlabeled_length]
unlabeled_dataset = Subset(train_dataset, indices)
# 为了模拟未标记的数据，我们创建一个新的数据集类，该类返回图像和伪标签
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, _ = self.subset[index]  # 忽略原始标签
        return x, torch.tensor(-1)  # 使用-1作为伪标签

    def __len__(self):
        return len(self.subset)

unlabeled_dataset = UnlabeledDataset(unlabeled_dataset)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f}\tAccuracy: {accuracy:.4f}')
    return train_loss, accuracy

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}\n')
    return test_loss, accuracy

# 主训练过程
def main():
    num_epochs = 10
    teacher = Teacher_model().to(device)
    student = Student_model().to(device)
    optimizer_teacher = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.5)
    optimizer_student = optim.SGD(student.parameters(), lr=0.01, momentum=0.5)

    teacher_train_losses = []
    teacher_train_accs = []
    student_train_losses = []
    student_train_accs = []

    # 先训练教师网络
    for epoch in range(1, num_epochs + 1):
        teacher_loss, teacher_acc = train(teacher, device, train_loader, optimizer_teacher, epoch)
        teacher_train_losses.append(teacher_loss)
        teacher_train_accs.append(teacher_acc)
    # 然后训练学生网络
    for epoch in range(1, num_epochs + 1):
        student_loss, student_acc = train(student, device, unlabeled_loader, optimizer_student, epoch)
        student_train_losses.append(student_loss)
        student_train_accs.append(student_acc)

    # 测试模型性能
    teacher_test_loss, teacher_test_acc = test(teacher, device, test_loader)
    student_test_loss, student_test_acc = test(student, device, test_loader)

    # 画图部分
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(teacher_train_losses, label='Teacher Loss')
    plt.plot(student_train_losses, label='Student Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(teacher_train_accs, label='Teacher Accuracy')
    plt.plot(student_train_accs, label='Student Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
