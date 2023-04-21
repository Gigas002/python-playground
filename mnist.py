import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#----------------------------------------------------------
num_epochs = 10         # Количество итераций циклов обучения
num_batch = 100         # Number of images to be processed at once
learning_rate = 0.001   # Коэффициент обучаемости
image_size = 28*28      # Разрешение изображения

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# Creating datasets for learning / evaluation

# Specification of conversion method
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.MNIST(
    './data',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # Conversion to tensor, etc.
    )

# 評価用
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)

test_dataloader = DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# Definition of neural network model
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

#----------------------------------------------------------
# Production of neural networks
model = Net(image_size, 10).to(device)

#----------------------------------------------------------
# Loss function settings
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# Setting of optimization method
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

#----------------------------------------------------------
# study
model.train()  # モデルを訓練モードにする

for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum : torch.Tensor = torch.zeros(1, dtype=torch.float32) # 0

    for inputs, labels in train_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Initialize Optimizer
        optimizer.zero_grad()

        # Process the neural network
        inputs = inputs.view(-1, image_size) # Change the image data part to one dimension
        outputs = model(inputs)

        # Calculation of loss (error between output and label)
        loss = criterion(outputs, labels)
        loss_sum += loss

        # Gradient calculation
        loss.backward()

        # Update weight
        optimizer.step()

    # Display of learning status
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # Save model weight
    # torch.save(model.state_dict(), 'model_weights.pth')

#----------------------------------------------------------
# evaluation
model.eval()  # Make the model into evaluation mode

loss_sum : torch.Tensor = torch.zeros(1, dtype=torch.float32) # 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # Get the correct answer value
        pred = outputs.argmax(1)
        # Count the correct answer
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
