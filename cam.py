import os
import cv2
import torch
import models
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from dataloader.dataset import edge_detection
from utils.common import config_loader, get_valid_args
   
class GradCAM:
    def __init__(self, model, target_layer, resize=True, return_edge=False, device=torch.device("cuda")):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.resize = resize
        self.return_edge = return_edge
        self.device = device

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    # PIL load image, same to training
    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((256, 256), Image.Resampling.LANCZOS) if self.resize else img
        return np.array(img)    
    
    def _preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_resized = self.load_image(image_path)
        image_tensor = transform(image_resized).unsqueeze(0).to(self.device)
        
        if self.return_edge:
            edge = edge_detection(image_resized)
            edge = transform(edge).unsqueeze(0).to(self.device)
            image_tensor = torch.cat((image_tensor, edge), dim=1)
        
        return image_tensor, image_resized

    def _generate_heatmap(self, gradients, activations):
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

    def _overlay_heatmap(self, image, heatmap):
        heatmap_resized = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        # Convert the heatmap from BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlayed_image = cv2.addWeighted(image, 1.0, heatmap_colored, 0.5, 0)
        return overlayed_image, heatmap_colored

    def generate(self, image_path, class_idx=None):
        image_tensor, image_resized = self._preprocess_image(image_path)
        image_tensor.requires_grad_()

        image_tensor = image_tensor.to(self.device)
        logits, _ = self.model(image_tensor)
        pred = logits.argmax(dim=1).item()
        if class_idx is None:
            class_idx = pred
        print(f"GT: {class_idx}, Pred: {pred}")

        self.model.zero_grad()
        logits[0, class_idx].backward()

        heatmap = self._generate_heatmap(self.gradients, self.activations)
        heatmap = heatmap.detach().cpu()

        overlayed_image, heatmap_colored = self._overlay_heatmap(image_resized, heatmap)
        return image_resized, overlayed_image, heatmap_colored

    def save_results(self, image_path, save_path, class_idx=None):
        original_image, overlayed_image, heatmap_colored = self.generate(image_path, class_idx)
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
        
        # 保存原始图像
        cv2.imwrite(os.path.join(save_path, 'original_image.jpg'), original_image)
        
        # 保存heatmap
        cv2.imwrite(os.path.join(save_path, 'heatmap.jpg'), heatmap_colored)
        
        # 保存叠加图像
        cv2.imwrite(os.path.join(save_path, 'overlayed_image.jpg'), overlayed_image)
        
        # 创建三张图像并排显示的拼接图
        combined_image = np.hstack((original_image, heatmap_colored, overlayed_image))
        cv2.imwrite(os.path.join(save_path, 'combined_image.jpg'), combined_image)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_type = "camera"
    data_type = "images"
    cig_brand = "1916"
    num_regions = 2
    cfg_path_dict = {
        "BaseModel": "./configs/base.yaml",
        "DFFD": "./configs/DFFD.yaml",
        "FaceEdge": "./configs/FaceEdge.yaml",
    }
    for model_name in cfg_path_dict.keys():
        return_edge = model_name == "FaceEdge"
        
        target_layer = "out_attention" if model_name == "DFFD" else "backbone.layer4.2"
        
        cfg_path = cfg_path_dict[model_name]
        cfg = config_loader(cfg_path)
        for region_id in range(num_regions):
            for label_id in range(2): # 1 for real, 0 for fake
                label_list = ["fake", "real"]
                label = label_list[label_id]
                class_idx = label_id * num_regions + region_id
                
                model_cfg = cfg['model_cfg']
                model_cfg['num_classes'] = num_regions * 2
                Model = getattr(models, model_cfg['type'])
                valid_args = get_valid_args(Model, model_cfg, ['type'])
                model = Model(**valid_args)
                model = model.to(device)
                
                ckpt_path = f"./outputs/{img_type}_data_256/{cig_brand}/{data_type}/CE/{model_name}/macro/logits/best_test_model.pt"
                model.load_state_dict(torch.load(ckpt_path)['model'])
                image_dir = f"./{img_type}_data_256/{cig_brand}/{data_type}/test/{region_id}/{label}/"
                img_list = os.listdir(image_dir)
                img_path = os.path.join(image_dir, img_list[0])
                
                grad_cam = GradCAM(model, target_layer=target_layer, resize=True, return_edge=return_edge, device=device)
                
                save_path = f"./grad_cam_results/{cig_brand}/{img_type}/{model_name}/{label}_{region_id}/"
                os.makedirs(save_path, exist_ok=True)
                grad_cam.save_results(img_path, save_path, class_idx=class_idx)
                
                if model_name == "BaseModel":
                    ckpt_path = f"./outputs/{img_type}_data_256/{cig_brand}/{data_type}/CE_AllContrastive/{model_name}/macro/logits/best_test_model.pt"
                    model.load_state_dict(torch.load(ckpt_path)['model'])
                    grad_cam = GradCAM(model, target_layer=target_layer, resize=True, return_edge=return_edge, device=device)
                
                    save_path = f"./grad_cam_results/{cig_brand}/{img_type}/{model_name}_regions/{label}_{region_id}/"
                    os.makedirs(save_path, exist_ok=True)
                    grad_cam.save_results(img_path, save_path, class_idx=class_idx)
                    
                