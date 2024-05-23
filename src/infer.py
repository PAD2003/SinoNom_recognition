from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import torch
import csv
from src.models.xla_module import XLALitModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.data.components.aug.wrapper_v2 import Augmenter
from src.data.components.aug.vietocr_aug import ImgAugTransform
from src.models.xla_module import XLALitModule

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# config
images_folder = "data/my_valid"
run_name = "2024-05-22_04-02-26"
checkpoint_path = "logs/train/runs/2024-05-22_04-02-26/checkpoints/last.ckpt"
manifest = "data/manifest_full.json"

output_path = "results/resnet18_noagument"


def get_decodevocab(manifest):
    with open(manifest, "r") as file:
        samples = json.load(file)["train"]
    
    keys = list(samples.keys())
    return dict(zip(range(len(keys)), keys))

def decode_labels(preds, decode_vocab):
    labels = []
    # print(decode_vocab)
    for pred in preds:
        pred = int(pred)
        label = decode_vocab[pred]
        labels.append(label)
    return labels

def save_csv(result_preds, csv_path):
    result_preds = {**{"image_name":"label"}, **result_preds}
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        for filename, class_ in result_preds.items():
            writer.writerow([filename, class_])

def save_npz(result_logits, npz_path):
    logits = result_logits.values()
    logits = np.stack(logits, axis=0)
    print("Save logits file: ", logits.shape)
    np.savez(npz_path, logits)

class XLAInferDataset(Dataset):
    def __init__(
        self,
        images_folder,
        base_augmenter = None,
        color_augmenter = None,
        val_p = None
    ):
        self.images_folder = images_folder
        self.base_augmenter = base_augmenter
        self.color_augmenter = color_augmenter
        self.val_p = val_p
        
        self.samples = self.load_data(images_folder)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        sample = self.samples[index]
        filename = sample['filename']
        image = sample['image']

        return {'filename': filename, 'image': image}

    def load_data(self, images_folder):
        samples = []
        for filename in os.listdir(images_folder):
            image_path = os.path.join(images_folder, filename)
            image = np.array(Image.open(image_path).convert("RGB"))
            if self.base_augmenter:
                image = self.base_augmenter.full_augment(image, 
                                                        choice=self.val_p,
                                                        fname=filename,
                                                        borderMode='native')
            if self.color_augmenter:
                image = self.color_augmenter(image)
            samples.append({'filename': filename, 'image': image})
        return samples

class XLAInferCollator(object):
    def __init__(self, num_class, image_shape=[64, 64]):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.num_class = num_class
        self.image_shape = image_shape

    def __call__(self, batch):
        filenames = []
        imgs = []
        for sample in batch:
            filename = sample['filename']
            image = sample['image']

            filenames.append(filename)
            imgs.append(self.transform(cv2.resize(image, 
                                                    self.image_shape, 
                                                    interpolation=cv2.INTER_LINEAR)))

        # print(labels)
        # print("imgs:",imgs[0].shape)
        rs = {
            'imgs': torch.stack(imgs, dim=0),
            'filenames': filenames
        }
        
        return rs

# @task_wrapper
def infer(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating agumenter")
    base_augmenter = None
    color_augmenter = None
    # if cfg.data.train_p != [0.0, 0.0, 0.0]:
    #     base_augmenter: Augmenter = hydra.utils.instantiate(cfg.data.base_augmenter)
    # color_augmenter: ImgAugTransform = hydra.utils.instantiate(cfg.data.color_augmenter)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = XLALitModule.load_from_checkpoint(checkpoint_path)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating data...")
    dataset = XLAInferDataset(
        images_folder=images_folder,
        base_augmenter=base_augmenter,
        color_augmenter=color_augmenter,
        val_p = cfg.data.val_p
    )
    collator = XLAInferCollator(num_class=cfg.model.num_classes,
                                image_shape=cfg.data.image_shape)
    dataloader = DataLoader(dataset = dataset,
                            batch_size = 2,
                            shuffle=False,
                            collate_fn=collator)
    
    log.info("Starting infering!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    decode_vocab = get_decodevocab(manifest)
    
    result_preds = {}
    result_logits = {}

    with torch.inference_mode():
        model.eval()
        for batch in dataloader:
            imgs = batch['imgs']
            imgs = imgs.to(device)
            filenames = batch['filenames']

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            labels = decode_labels(preds, decode_vocab)
            
            print(filenames, logits)
            
            for i, filename in enumerate(filenames):
                result_preds[filename.split('.')[0]] = int(labels[i])
                result_logits[filename.split('.')[0]] = np.array(logits[i].cpu())
    
    result_preds = dict(sorted(result_preds.items()))
    result_logits = dict(sorted(result_logits.items()))
    
    try:
        os.mkdir(path=output_path)
    except OSError as error:
        print(error)
        # return
    
    save_npz(result_logits, os.path.join(output_path, "logits.npz"))
    save_csv(result_preds, os.path.join(output_path, "preds.csv"))

@hydra.main(version_base="1.3", config_path=f"../logs/train/runs/{run_name}/.hydra", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    infer(cfg)


if __name__ == "__main__":
    main()
