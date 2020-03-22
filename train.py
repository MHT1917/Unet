import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import UNet
import MKDataset

path = r'F:\DataSets\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
module = r'module.pkl'
img_save_path = r'F:\train_img'
batch = 1

net = UNet.MainNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=4, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print('module is loaded !')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

while True:
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()

        xs_ = net(xs)

        loss = loss_func(xs_, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('batch: {}  count: {}  loss: {}'.format(batch, (i + 1) * 4, loss))
            # x = xs[0]
            # x_ = xs_[0]
            # y = ys[0]
            # z = torch.cat((x, x_, y), 2)
            # img_save = transforms.ToPILImage()(z.cpu())
            # img_save.save(os.path.join(img_save_path, '{}.png'.format(batch)))
    torch.save(net.state_dict(), module)
    print('module is saved !')
    batch += 1