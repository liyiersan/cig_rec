import os
import torch
import models
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.common import config_loader, get_valid_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path):
    img = Image.open(image_path)
    image_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_resized).unsqueeze(0).to(device)
    return image_tensor

def load_model(cfg_path, model_path):
    cfg = config_loader(cfg_path)
    model_cfg = cfg['model_cfg']
    model_cfg['num_classes'] = 4
    Model = getattr(models, model_cfg['type'])
    valid_args = get_valid_args(Model, model_cfg, ['type'])
    model = Model(**valid_args)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def load_prototypes(prototypes_path, threshold_path):
    prototypes = torch.load(prototypes_path)
    thresholds = torch.load(threshold_path)
    return prototypes, thresholds

def logits_predict(model, image_tensor):
    with torch.no_grad():
        logits, _ = model(image_tensor)
        pred = torch.argmax(logits, dim=1)
        pred = pred.item() > 1
    return pred

def metric_predict(model, image_tensor, propotypes, thresholds):
    with torch.no_grad():
        _, feats = model(image_tensor)
        dists = torch.cdist(feats, propotypes)
        min_dists, min_index = torch.min(dists, dim=1)
        is_real = min_dists < thresholds[min_index].squeeze(0)
    return is_real.item()

if __name__ == '__main__':
    cfg_path = 'configs/base.yaml'
    data_type = 'camera'
    model_path = f'ckpts/{data_type}_1916/model.pt'
    prototypes_path = f'ckpts/{data_type}_1916/prototypes.pt'
    thresholds_path = f'ckpts/{data_type}_1916/thresholds.pt'
    image_dir = f'data/test/{data_type}_1916/'

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            if "fake" in image_path:
                gt = "fake"
            else:
                gt = "real"
            model = load_model(cfg_path, model_path)
            prototypes, thresholds = load_prototypes(prototypes_path, thresholds_path)
            image_tensor = load_image(image_path)

            pred_logits = "real" if logits_predict(model, image_tensor) else "fake"
            pred_metric = "real" if metric_predict(model, image_tensor, prototypes, thresholds) else "fake"

            print(f'Image Path: {image_path}')
            print(f'Ground Truth: {gt}, Logits Prediction: {pred_logits}, Metric Prediction: {pred_metric}')
