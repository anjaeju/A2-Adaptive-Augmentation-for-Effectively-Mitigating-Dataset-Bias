import argparse, os, sys, math
from glob import glob
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision import utils

import lpips
from model import Generator

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def denormalize(x):
    mean = np.asarray([ 0.5, 0.5, 0.5 ])
    std = np.asarray([ 0.5, 0.5, 0.5 ])
    denorm = transforms.Normalize((-1 * mean / std), (1.0 / std))
    return denorm(x)

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def make_image_v2(y, y_hat):
    y     = y.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).permute(0, 2, 3, 1).to("cpu")
    y_hat = y_hat.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).permute(0, 2, 3, 1).to("cpu")
    return torch.cat([y, y_hat], dim=2)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Image projector to the generator latent spaces")
    
    parser.add_argument("--lr_rampup",type=float,default=0.05,help="duration of the learning rate warmup",)
    parser.add_argument("--lr_rampdown",type=float,default=0.25,help="duration of the learning rate decay",)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--noise", type=float, default=0.05, help="strength of the noise level")
    parser.add_argument("--noise_ramp",type=float,default=0.75,help="duration of the noise level decay",)
    
    parser.add_argument("--noise_regularize",type=float,default=1e5,help="weight of the noise regularization",)
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument("--w_plus",action="store_true",help="allow to use distinct latent codes to each layers",)
    
    parser.add_argument("--step", type=int, default=3000, help="optimize iterations")
    parser.add_argument("--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--size", type=int, default=256, help="output image sizes of the generator")
    parser.add_argument("--file_path", type=str, help="path to image files to be projected")
    parser.add_argument("--save_path", type=str, help="path to image files to be saved")
    parser.add_argument("--num_images", type=int, help="# of Batch size for projection")

    args = parser.parse_args()

    n_mean_latent = 10000

    # resize = min(args.size, args.size)

    transform = transforms.Compose([
            transforms.Resize((args.size, args.size)),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    os.makedirs(args.save_path, exist_ok=True)

    # Dataset Preparation
    args.files = sorted([y for x in os.walk(args.file_path) for y in glob(os.path.join(x[0], '*.png'))])
    # batch = len(args.files)//args.num_images
    # splited_arrays = np.array_split(args.files, batch)

    splited_arrays = args.files
    
    # splited_arrays = [args.files]
    print(f'Num of files: {len(args.files)} -> batch {args.num_images} of {len(splited_arrays)} lists')

    # Model Generation
    if True:
        g_ema = Generator(args.size, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        g_ema = torch.nn.DataParallel(g_ema)
        g_ema.eval()
        g_ema = g_ema.to(device)
    else:
        g_ema = Generator(args.size, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)

    result_file = {}
    for img_idx, paths in enumerate(splited_arrays):
        # label_y = [transform(Image.open(each_path).convert('RGB')) for each_path in paths]
        # label_y = torch.stack(label_y, 0).to(device)    # B x 3 x size x size

        label_y = transform(Image.open(paths).convert('RGB')).cuda()

        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            # latent_out = g_ema.style(noise_sample)
            latent_out = g_ema.module.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

        # noises_single = g_ema.make_noise()
        noises_single = g_ema.module.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(label_y.shape[0], 1, 1, 1).normal_())
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(label_y.shape[0], 1)

        if args.w_plus:
            # latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.module.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
            batch, channel, height, width = img_gen.shape

            # if height > 256:
            #     factor = height // 256

            #     img_gen = img_gen.reshape(
            #         batch, channel, height // factor, factor, width // factor, factor
            #     )
            #     img_gen = img_gen.mean([3, 5])

            #     label_y = label_y.reshape(
            #         batch, channel, height // factor, factor, width // factor, factor
            #     )
            #     label_y = label_y.mean([3, 5])

            p_loss = percept(img_gen, label_y).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, label_y)
            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)
            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description((
                f"{img_idx}/{len(splited_arrays)}"
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )


        recon_y, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)   # B x 3 x size x size

        grid_img = torch.cat([label_y.unsqueeze(0), recon_y], dim=0)
        
        # grid_img = torch.cat([ torch.stack([each_label, each_recon], dim=0) for (each_label, each_recon) in zip(label_y, recon_y) ])
        utils.save_image(grid_img,f'{args.save_path}/{img_idx}.png', nrow=4, normalize=True, range=(-1, 1),)

        for each_latent, each_recon, each_label, each_path in zip(latent_path[-1], recon_y, label_y, paths):
            
            # each_latent -> 1 x 512
            key_name = each_path.split('/')[-1]            
            result_file[key_name] = {"latent": each_latent} #,"noise": noise_single,}

            torch.save(result_file, f'{args.save_path}/projected.pt')
            # recon_array = make_image(each_recon.unsqueeze(0))    # 1 x 256 x 256 x 3
            # label_array = make_image(each_label.unsqueeze(0))    # 1 x 256 x 256 x 3

            # img_name1 = f'{args.save_path}/{key_name}-generated.png'
            # img_name2 = f'{args.save_path}/{key_name}-label.png'
            # Image.fromarray(recon_array).save(img_name1)
            # Image.fromarray(label_array).save(img_name1)

            

        

        
        
