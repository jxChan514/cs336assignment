import numpy as np  # 处理 token ID 数组，并用 np.random.randint 抽取起点
import torch  # 将输入和 targets 转为 torch.long 张量，放到指定设备

def get_batch(dataset, batch_size, context_length, device):
    n=dataset.size
    start=np.random.randint(0, high=n-context_length, size=batch_size, dtype=int)
    idx = start[:, None] + np.arange(context_length)
    train_data=dataset[idx]
    ground_truth=dataset[idx+1]
    inputs=torch.tensor(train_data,dtype=torch.long,device=device)
    targets=torch.tensor(ground_truth,dtype=torch.long,device=device)
    return inputs,targets