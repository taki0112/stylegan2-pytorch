import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import torch.distributed as dist
import math
from tqdm import tqdm
from torchvision import transforms
from scipy import linalg
import pickle, os

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)

def extract_real_feature(data_loader, inception, device):
    feats = []

    for img in tqdm(data_loader):
        img = img.to(device)
        feat = inception(img)

        feats.append(feat)

    feats = gather_feats(feats)

    return feats

def normalize_fake_img(imgs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imgs = (imgs + 1) / 2  # -1 ~ 1 to 0~1
    imgs = torch.clamp(imgs, 0, 1, out=None)
    imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear")
    imgs = transforms.Normalize(mean=mean, std=std)(imgs)

    return imgs

def gather_feats(feats):
    feats = torch.cat(feats, dim=0)
    feats = torch.cat(GatherLayer.apply(feats), dim=0)
    feats = feats.detach().cpu().numpy()

    return feats

def extract_fake_feature(generator, inception, num_gpus, device, latent_dim, fake_samples=50000, batch_size=16):
    num_batches = int(math.ceil(float(fake_samples) / float(batch_size * num_gpus)))
    feats = []
    for _ in tqdm(range(num_batches)):
        z = [torch.randn([batch_size, latent_dim], device=device)]
        fake_img = generator(z)

        fake_img = normalize_fake_img(fake_img)

        feat = inception(fake_img)

        feats.append(feat)

    feats = gather_feats(feats)

    return feats

def get_statistics(feats):
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)

    return mu, cov

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

@torch.no_grad()
def calculate_fid(data_loader, generator_model, inception_model, dataset_name, rank, device,
                  latent_dim, fake_samples=50000, batch_size=16):

    num_gpus = torch.cuda.device_count()

    generator_model = generator_model.eval()
    inception_model = inception_model.eval()

    pickle_name = '{}_mu_cov.pickle'.format(dataset_name)
    cache = os.path.exists(pickle_name)

    if cache:
        with open(pickle_name, 'rb') as f:
            real_mu, real_cov = pickle.load(f)
    else:
        real_feats = extract_real_feature(data_loader, inception_model, device=device)
        real_mu, real_cov = get_statistics(real_feats)

        if rank == 0:
            with open(pickle_name, 'wb') as f:
                pickle.dump((real_mu, real_cov), f, protocol=pickle.HIGHEST_PROTOCOL)


    fake_feats = extract_fake_feature(generator_model, inception_model, num_gpus, device, latent_dim, fake_samples, batch_size)
    fake_mu, fake_cov = get_statistics(fake_feats)

    fid = frechet_distance(real_mu, real_cov, fake_mu, fake_cov)
    return fid




