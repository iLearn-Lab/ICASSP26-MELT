import torch


a_vct=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
print(a_vct.shape)
b=torch.randn_like(a_vct)  # 1. 正常高斯噪声
print(b)
c=a_vct.sort(dim=1, descending=True)
print(c)
d=b.sort(dim=1, descending=True)
print(d)
