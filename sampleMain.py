import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from Modules.LossFunction import *
from Modules.Optimziers import *




device = torch.device("cuda:0")


''' 
---------------------------------------------------------------------------------------------------------
# x = torch.tensor(~~)
# x2 = x.to(device)
#.grad 는 맨 종단인 (leaf) x에 사용할 수 있다.  다시 할당된 x2 는 leaf 가 아니기 때문에, .grad 를 사용할 수 없다.

x = torch.tensor([[1.0,2.0,3.0],[3.0,4.0,6.0]])
x.requires_grad = True

x2 = x.to(device)

y = x2 ** 2 + 3

target = torch.tensor([[10.0,10.0,10.0],[10.0,10.0,10.0]])
target = target.to(device)
loss = torch.sum(torch.abs(y-target))
loss.backward()

print(x.grad)
---------------------------------------------------------------------------------------------------------
'''

'''
x = init.uniform_(torch.Tensor(6,6),-10,10)
noise = init.normal_(torch.Tensor(6,6),std=1)

y = 2*x + 3
y_noise = y + noise

model = nn.Linear(6,6)
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

label = y_noise

for i in range(1) :
    optimizer.zero_grad() #기울기 0 초기화
    param_list = list(model.parameters())
    w_t,b_t = param_list[0], param_list[1]
    print(param_list[0], param_list[1])  # w , b
    output = model(x)

    loss = loss_func(output,label)
    loss.backward()  # 각 변수별 기울기 계산.
    optimizer.step() # weight 조정.

    if i % 10 == 0 :
        print(loss.data)

    gradient = param_list[0].grad
    print(gradient)           # 그래디언트 // gradient

    param_list = list(model.parameters())
    w_t2 ,w_b2 = param_list[0], param_list[1]  # w_t2 = w_t1 - gradient * LR
    print(param_list[0],param_list[1]) # w , b
---------------------------------------------------------------------------------------------------------
'''


'''
---------------------------------------------------------------------------------------------------------


num_data = 50
num_epoch = 150

x = init.uniform_(torch.Tensor(num_data,1),-10,10)
noise = init.normal_(torch.Tensor(num_data,1),std=1)
y = x**2 + 3
y_noise = y + noise

model = nn.Sequential(
    nn.Linear(1,6),
    nn.ReLU(),
    nn.Linear(6,10),
    nn.ReLU(),
    nn.Linear(10,20),
    nn.ReLU(),
    nn.Linear(20,1)
)

loss_func = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(),lr=0.001)

loss_array = []

for i in range(num_epoch) :
    optimizer.zero_grad()
    output = model(x)
    loss_func(output,y_noise).backward()
    optimizer.step()

    loss_array.append(loss_func(output,y_noise))

print(loss_array)

import matplotlib.pyplot as plt
plt.plot(loss_array)
plt.show()

---------------------------------------------------------------------------------------------------------
'''


