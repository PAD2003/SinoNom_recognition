import os
import numpy as np
import json
import csv
import path
import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

def get_logits(results_folder):
    all_logits = {}
    for root, dirs, files in os.walk(results_folder):
        
        if root == results_folder:
            continue
        model = root.split("/")[-1]
        logits = np.load(os.path.join(root, "logits.npz"))
        all_logits[model] = logits['arr_0']
        
    return all_logits

def get_filenames(folder_path):
    
    folder = path.Path(folder_path)
    jpg_files = list(folder.glob('*.jpg'))
    sorted_files = sorted(jpg_files, key=lambda x: int(x.stem))

    return [str(file.name) for file in sorted_files]

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
        
        for filename, class_id in result_preds.items():
            writer.writerow([filename, class_id])

def soft_ensemble(all_logits, weights, manifest, images_folder):
    print(weights)
    # assert sum(weights) == 1
    ensemble_logits = np.zeros(shape=list(all_logits.values())[0].shape)
    for i, (model, logits) in enumerate(all_logits.items()):
        ensemble_logits += logits * weights[i]
    
    ensemble_preds = np.argmax(ensemble_logits, axis=1)
    decode_vocab = get_decodevocab(manifest)
    ensemble_labels = decode_labels(ensemble_preds, decode_vocab)
    
    filenames = get_filenames(images_folder)
    results = {}
    assert len(filenames) == len(ensemble_labels)
    for filename, ensemble_label in zip(filenames, ensemble_labels):
        results[filename.split('.')[0]] = int(ensemble_label)
    
    return ensemble_logits, results

@hydra.main(version_base="1.3", config_path="../configs", config_name="ensemble.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    results_folder = cfg.results_folder
    manifest = cfg.manifest
    images_folder = cfg.images_folder
    
    all_logits = get_logits(results_folder)
    num_models = len(all_logits)
    ensemble_logits, results = soft_ensemble(all_logits, [float(1/num_models)]*num_models, manifest, images_folder)

    save_csv(results, f"{results_folder}/ensemble.csv")


if __name__ == "__main__":
    main()
