import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 初始化数据集并参考
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))


'''
我们首先定义一个模型变量net，它是一个Sequential类的实例。 
Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 
然后将第一层的输出作为第二层的输入，以此类推。 在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。 
但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”。'''
# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
number_epochs = 3
for epoch in range(number_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')