import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import random
import wandb
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import configargparse

import time
import math

def psnr(ori_img, pred_img):
  max_pixel = 1.0

  mse = ((ori_img-pred_img)**2).mean()

  if(mse == 0):
    return 100

  psnr = 20* math.log10(max_pixel / math.sqrt(mse))

  return psnr

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def free_image_tensor(sidelength, directory):
    img = cv2.imread(directory)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = srgb_to_linear(img)
    print(img)
    plt.imshow(img)
    plt.savefig('./linear.png')

    transform = Compose([
              ToTensor()
            ])
    img = transform(img)
    print(img.shape)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength, directory):
        super().__init__()
        img = free_image_tensor(sidelength, directory)
        self.length = sidelength * sidelength
        self.pixels = img.permute(1, 2, 0).view(-1, 3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

class ImageFitting_batch(Dataset):
    def __init__(self, sidelength, directory, batch):
        super().__init__()
        img = free_image_tensor(sidelength, directory)
        self.length = sidelength * sidelength
        self.pixels = img.permute(1, 2, 0).view(-1, 3)
        self.coords = get_mgrid(sidelength, 2)
        self.batch_list = list(range(self.length))
        self.batch_size = batch
        self.len = self.length / self.batch_size

    def __len__(self):
        return self.length

    def getitem(self, idx):
        
        return self.coords[idx*self.batch_size:(idx+1)*self.batch_size], self.pixels[idx*self.batch_size:(idx+1)*self.batch_size]

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class Image_NGP(torch.nn.Module):
    def __init__(self, filename, device):
        self.data = read_image(filename)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

def read_image(file):
    img = read_image_imageio(file)
    img = srgb_to_linear(img)
    
    return img

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=18)
p.add_argument('--lr', type=float, default=-10, help='learning rate. default=2**-10')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--sidelength', type=int, default=512)
p.add_argument('--num_workers', type=int, default=0)
p.add_argument('--project', type=str, default ="Mars_18", help = 'Project name')
p.add_argument('--max_steps', type=int, default = 10000)
p.add_argument('--directory', type=str)
p.add_argument('--gpu_num', type=str, default = "0")
opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num

loaded_data = ImageFitting(opt.sidelength, opt.directory)
#loaded_data2 = ImageFitting_batch(opt.sidelength, opt.directory, 2**opt.batch_size)
full_loader = DataLoader(loaded_data, batch_size=2**18, shuffle=False, pin_memory=True, num_workers=opt.num_workers)
dataloader = MultiEpochsDataLoader(loaded_data, batch_size=2**opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.num_workers)

img_siren = Siren(in_features=2, out_features=3, hidden_features=128,
                  hidden_layers=3, outermost_linear=True)
img_siren.cuda()

# Train
run = wandb.init(
    # set the wandb project where this run will be logged
    entity = "opt_bs",
    project=opt.project,
    name = opt.experiment_name,
    tags = [opt.project, "batch_size="+str(opt.batch_size), "learning_rate="+str(opt.lr), "half"],
    # track hyperparameters and run metadata
    config={
    "learning_rate": opt.lr,
    "dataset": "Mars18",
    "batch_size" : 2**opt.batch_size
    }
)

optim = torch.optim.Adam(lr=2**opt.lr, params=img_siren.parameters())

model_full, ground_full = next(iter(full_loader))
model_full, ground_full = model_full.cuda(), ground_full.cuda()

#plt.imshow(ground_truth.cpu().view(sidelength, sidelength, 3).detach().numpy())

steps_til_summary = opt.steps_til_summary
step = 0
optim.zero_grad()
end = 1
check_freq = 1
max_steps = opt.max_steps
while end:
    for batch_idx, (model_input, ground_truth) in enumerate(dataloader):
        
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth)**2).mean()


        if (not step % steps_til_summary) or check_freq:
            img_siren.eval()
            model_test, coords_test = img_siren(model_full)
            Test_loss = ((model_test - ground_full)**2).mean()
            img_siren.train()


            ps = psnr (ground_full, model_test)
            print("step: %d, PSNR: %0.4f, TestLoss: %0.4f" % (step, ps, Test_loss))
            wandb.log({"step": step, "PSNR": ps, "loss": loss, "Test_loss": Test_loss})

            
            if(ps > 37): check_freq = 1
        
            if(ps > 50):
                print('Train steps= ', step)
                plt.imshow(model_test.cpu().view(opt.sidelength, opt.sidelength, 3).detach().numpy())
                plt.savefig('./1.png')
                end = 0
                break

        optim.zero_grad()
        loss.backward()
        optim.step()

        if(step >= max_steps):
            end = 0
            break
        step+=1
    
        

wandb.finish()
