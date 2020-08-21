#ライブラリのインポート
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import torch.optim as optim

def 

#パラメータ
epochs=50
batchsize=100

#画像の前処理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

#訓練画像の用意(mnist)
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
#テスト画像の用意(mnist)
testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
#画像の確認
print("訓練画像数",len(trainset))
print("テスト画像数",len(testset))
    
#訓練画像のローダーの用意
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batchsize,
                                            shuffle=True,
                                            num_workers=2)
#テスト画像のローダーの用意
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batchsize,
                                            shuffle=False, 
                                            num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

#ネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)
        
        self.mean = nn.Linear(10,2) #latent
        self.var = nn.Linear(10,2)#latent
        
        self.fc3 = nn.Linear(2, 10)
    def block1(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        return x
    
    def block2(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return x
    def Latent0(self,x):
        z_mean =self.mean(x)
        z_var =self.var(x)
        return z_mean,z_var
    
    def Latentz(self,z_mean,z_var):
        std = z_var.exp().sqrt()
        epsilon = torch.randn(2, device=device)
        z = z_mean + std * epsilon
        return z
        
        

    def forward(self, x):
        x = self.block1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = self.block2(x)
        z_mean,z_var = self.Latent0(x)
        std = z_var.exp().sqrt()
        epsilon = torch.randn(2, device=device)
        x = z_mean + std * epsilon
        
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x
    
    
#GPUがある場合はGPUを利用
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

net = Net().to(device)

#最適化と損失関数の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



#学習
train_loss_hi=[]
train_acc_hi=[]

for epoch in range(epochs):
    train_loss,train_acc = 0.0,0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        #画像とラベルをGPUへ転送
        inputs,labels = inputs.to(device),labels.to(device)
        # 勾配の初期化
        optimizer.zero_grad()

        # 画像をモデルへ入力
        outputs = net(inputs)
        #精度の計算
        pred = outputs.max(1)[1]
        acc=(pred == labels).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        train_acc += acc.item()
        
        #イテレーション毎の計算
        # if i % 100 == 99:
        #     print('epoch数:{:d}, イテレーション数:{:4d} ,train_acc:{:3f},train_loss: {:.3f}'
        #           .format(epoch + 1, i + 1, train_acc/batchsize,train_loss /batchsize))
        #     train
        #     train_acc_hi.append(train_acc/batchsize)
        #     train_loss = 0.0
        #     train_acc = 0.0
    print('epoch数:{:d},train_acc:{:3f},train_loss: {:.3f}'.format(epoch + 1, train_acc/len(trainloader),train_loss /len(trainloader)))
    train_acc_hi.append(train_acc/batchsize)
    train_loss_hi.append(train_loss/batchsize)
print('Finished Training')


#学習結果の確認
correct = 0
total = 0
cmp = plt.get_cmap("tab10")
plt.figure()

with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        x1=net.block1(images)
        x1 = x1.view(-1, 12 * 12 * 64)
        x1=net.block2(x1)
        z_mean,z_var=x1=net.Latent0(x1)
        std = z_var.exp().sqrt()
        epsilon = torch.randn(2, device=device)
        z = z_mean + std * epsilon
        
        #潜在空間zの可視化
        for i in range(0,10):
            select_flag = predicted == i
            plt_latent = z[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".",label=str(i))
# plt.legend()
plt.title("plot z space")
plt.savefig('latent_z_space.png')
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
axL.plot(range(epochs), train_acc_hi,
         color='blue', linestyle='-', label='train_acc')
axL.legend()
axL.set_xlabel('epoch')
axL.set_ylabel('train_acc')
axL.set_title('Training Accuracy')
axL.grid()

axR.plot(range(epochs), train_loss_hi,
         color='blue', linestyle='-', label='train_loss')
axR.legend()
axR.set_xlabel('epoch')
axR.set_ylabel('train_loss')
axR.set_title('Training Loss')
axR.grid()

fig.savefig('train_data.png')

#z plot
