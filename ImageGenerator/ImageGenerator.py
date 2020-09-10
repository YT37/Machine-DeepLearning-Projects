import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import (
    BatchNorm2d,
    BCELoss,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader = DataLoader(
    CIFAR10(
        root="./Data",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    ),
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

try:
    os.mkdir("./Models")
    os.mkdir("./Results")

except FileExistsError:
    pass


class Gen(Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.main = Sequential(
            ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            BatchNorm2d(512),
            ReLU(True),
            ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            ReLU(True),
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            ReLU(True),
            ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            BatchNorm2d(64),
            ReLU(True),
            ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            Tanh(),
        )

    def forward(self, data):
        output = self.main(data)
        return output


class Dis(Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.main = Sequential(
            Conv2d(3, 64, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(256, 512, 4, 2, 1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(512, 1, 4, 1, 0, bias=False),
            Sigmoid(),
        )

    def forward(self, data):
        output = self.main(data)
        return output.view(-1)


def weights(obj):
    classname = obj.__class__.__name__
    if classname.find("Conv") != -1:
        obj.weight.data.normal_(0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        obj.weight.data.normal_(1.0, 0.02)
        obj.bias.data.fill_(0)


gen = Gen()
gen.apply(weights)
gen.to(device)

dis = Dis()
dis.apply(weights)
dis.to(device)

criterion = BCELoss()

optimizerDis = Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerGen = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(25):
    for batch, dataset in enumerate(tqdm(dataloader, total=len(dataloader))):
        dis.zero_grad()

        data = dataset[0].to(device)

        target = torch.ones(data.size()[0], device=data.device)
        output = dis(data)

        realError = criterion(output, target)

        noise = torch.randn(data.size()[0], 100, 1, 1, device=data.device)
        fake = gen(noise)
        target = torch.zeros(data.size()[0], device=data.device)
        output = dis(fake.detach())

        fakeError = criterion(output, target)

        errD = realError + fakeError
        errD.backward()
        optimizerDis.step()

        gen.zero_grad()

        target = torch.ones(data.size()[0], device=data.device)
        output = dis(fake)

        errG = criterion(output, target)
        errG.backward()
        optimizerGen.step()

        print(f"  {epoch+1}/25 Dis Loss: {errD.data:.4f} Gen Loss: {errG.data:.4f}")

save_image(dataset[0], "./Results/Real.png", normalize=True)
save_image(gen(noise).data, f"./Results/Fake{epoch+1}.png", normalize=True)
torch.save(gen, f"./Models/model{epoch+1}.pth")
