import os
import numpy as np
import json
import csv

# config
results_folder = "results"
manifest = "data/manifest_full.json"
images_folder = "data/wb_recognition_dataset/val/images"

# code
def get_logits(results_folder):
    all_logits = {}
    for root, dirs, files in os.walk(results_folder):
        
        if root == results_folder:
            continue
        print(root)
        model = root.split("/")[-1]
        logits = np.load(os.path.join(root, "logits.npz"))
        all_logits[model] = logits['arr_0']
        
    return all_logits

def get_filenames(images_folder):
    return sorted(os.listdir(images_folder))

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

def soft_ensemble(all_logits, weights):
    assert sum(weights) == 1
    ensemble_logits = np.zeros(shape=list(all_logits.values())[0].shape)
    for i, (model, logits) in enumerate(all_logits.items()):
        ensemble_logits += logits * weights[i]
    
    ensemble_preds = np.argmax(ensemble_logits, axis=1)
    decode_vocab = get_decodevocab(manifest)
    ensemble_labels = decode_labels(ensemble_preds, decode_vocab)
    
    filenames = get_filenames(images_folder)
    results = {}
    for filename, ensemble_label in zip(filenames, ensemble_labels):
        results[filename.split('.')[0]] = int(ensemble_label)
    
    return ensemble_logits, results

all_logits = get_logits(results_folder)
ensemble_logits, results = soft_ensemble(all_logits, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
save_csv(results, f"{results_folder}/ensemble.csv")