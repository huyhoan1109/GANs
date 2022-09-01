""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    EMA,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config

torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            #transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    disc,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_disc,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_disc,
    ema,
):
    loop = tqdm(loader, leave=True)
    gen2 = Generator(
        config.Z_DIM, config.W_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, config.Z_DIM).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            disc_real = disc(real, alpha, step)
            disc_fake = disc(fake.detach(), alpha, step)
            gp = gradient_penalty(disc, real, fake, alpha, step, device=config.DEVICE)
            loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(disc_real ** 2))
            )

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Generator: max E[disc(gen_fake)] <-> min -E[disc(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % 100 == 0:
            ema(gen)
            with torch.no_grad():
                ema.copy_weights_to(gen2)
                fixed_fakes = gen2(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_disc.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_disc=loss_disc.item(),
        )


    return tensorboard_step, alpha


def main():
    gen = Generator(
        config.Z_DIM, config.W_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    disc = Discriminator(
        config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    ema = EMA(gamma=0.999, save_frequency=2000)
    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                          {"params": gen.map.parameters(), "lr": 1e-5}], lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_disc, disc, opt_disc, config.LEARNING_RATE,
        )

    gen.train()
    disc.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5   # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                disc,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_disc,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
                ema,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_disc)

        step += 1  # progress to the next img size


if __name__ == "__main__":
    main()