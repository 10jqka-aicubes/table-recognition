import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import numpy as np
import argparse
from losses import dice_coeff
from model.UNet.unet_model import UNet

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    for batch in loader:
        imgs, true_masks = batch["image"], batch["mask"]
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        pred = torch.sigmoid(mask_pred)
        pred1 = (pred > 0.5).float()
        tot += dice_coeff(pred1, true_masks).item()

    net.train()
    return tot / n_val


def train_net(net, device, img_size, epochs, batch_size, lr, path_paras, save_cp=True):
    train = BasicDataset(path_paras["train_dir_img"], path_paras["train_dir_mask"], img_size, mask_suffix="_")
    val = BasicDataset(path_paras["val_dir_img"], path_paras["val_dir_mask"], img_size, mask_suffix="_")

    n_train = len(train)
    n_val = len(val)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    global_step = 0
    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    """
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.6, patience=3)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            val_score = 0
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

                    logging.info(
                        "learning_rate: {}, Validation Dice Coeff: {}".format(
                            optimizer.param_groups[0]["lr"], val_score
                        )
                    )

        if save_cp:
            try:
                os.mkdir(path_paras["dir_checkpoint"])
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(
                net.state_dict(),
                os.path.join(path_paras["dir_checkpoint"], f"CP_epoch{epoch + 1}_score_{val_score}.pth"),
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--epochs", metavar="E", type=int, default=50, help="Number of epochs", dest="epochs")
    parser.add_argument(
        "-b", "--batch_size", metavar="B", type=int, nargs="?", default=1, help="Batch size", dest="batchsize"
    )
    parser.add_argument(
        "-l", "--learning_rate", metavar="LR", type=float, nargs="?", default=0.001, help="Learning rate", dest="lr"
    )
    parser.add_argument(
        "-d", "--data_dir", metavar="D", type=str, nargs="?", help="Path of train data", dest="dataDir", required=True
    )
    parser.add_argument(
        "-s",
        "--save_model_dir",
        metavar="S",
        type=str,
        nargs="?",
        help="Model save path",
        dest="savePath",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    data_dir = args.dataDir
    path_paras = {
        "train_dir_img": os.path.join(data_dir, "train/imgs/"),
        "train_dir_mask": os.path.join(data_dir, "train/gt/"),
        "val_dir_img": os.path.join(data_dir, "val/imgs/"),
        "val_dir_mask": os.path.join(data_dir, "val/gt/"),
        "dir_checkpoint": args.savePath,
    }

    epochs = args.epochs
    batchsize = args.batchsize
    lr = args.lr
    img_size = (640, 640)

    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling\n'
    )

    net.to(device=device)

    try:
        train_net(
            net=net, img_size=img_size, epochs=epochs, batch_size=batchsize, lr=lr, device=device, path_paras=path_paras
        )

    except KeyboardInterrupt:
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
