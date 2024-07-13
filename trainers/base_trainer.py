import os
import torch
import numpy as np
import os.path as osp
import torch.optim as optim
import models
import dataloader.dataset as dataset
from torch.utils.data import DataLoader
from losses.base import TrainingLoss
from utils.common import get_valid_args, ts2np
from utils.utils import cal_metrics, get_confusion_matrix, plot_confusion_matrix, plot_features, plot_loss


class BaseTrainer():
    """
    训练器的基类
    """
    def __init__(self, cfgs):
        """
            依据配置文件构建训练器, 包括: 模型, 数据加载器, 损失函数, 优化器, 学习率调整器, 训练参数等

            下面说明一下BaseTrainer的设计逻辑:
                1, 通过配置文件构建训练器, 配置文件中包含了训练器的所有参数, 包括: 模型, 数据加载器, 损失函数, 优化器, 学习率调整器, 训练参数等
                2, train()方法用于训练模型, 训练过程中会在测试集上进行测试, 并保存最好的模型.
                3, test()方法用于在测试集上测试模型, 并可视化混淆矩阵和特征. 
                5, save()方法用于保存模型.
                6, load()方法用于加载模型. 

            关于测试过程的说明:
                1, 测试时有两种策略获得预测结果. 
                2, 基于分类器输出的概率, 通过设置阈值, 将概率大于阈值的样本认为是目标类别, 否则认为是未知类别. 
                3, 基于特征, 通过计算特征的相似度, 将相似度大于阈值的样本认为是目标类别, 否则认为是未知类别.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.trainer_cfg = cfgs['trainer_cfg']

        # 创建数据加载器
        self.train_loader, self.test_loader, self.train_loader_val = self.build_loaders(cfgs['data_cfg'], cfgs['dataloader_cfg'])
        
        self.model = self.build_model(cfgs['model_cfg'])
        self.model_type = cfgs['model_cfg']['type']  
        
        # 创建损失函数, 优化器, 学习率调整器
        self.criterion = TrainingLoss(cfgs['loss_cfg'])
        self.optimizer = self.get_optimizer(cfgs['optimizer_cfg'])
        self.lr_scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.loss_type = "Binary" if "Binary" in cfgs['data_cfg']['type'] else "All"
        self.avg_type = "binary" if self.loss_type == "Binary" else "macro"

        # 训练有关参数
        self.start_epoch = 0
        # 最佳测试指标, 用于保存最好的模型
        self.best_test_metric = self.trainer_cfg['best_test_metric']
        self.epochs = self.trainer_cfg['epochs']
        
        # 早停参数
        self.early_stop = self.trainer_cfg['early_stop'] 
        
        # 推理相关
        self.inference_type = self.trainer_cfg['inference_type']
        self.alpha = self.trainer_cfg['alpha']
        
        self.num_classes = self.model.num_classes
        
        loss_dir = ""
        for i, loss in enumerate(self.criterion.loss_list):
            loss_name = loss[:-4]
            loss_dir += loss_name
            if i != len(self.criterion.loss_list) - 1:
                loss_dir += "_"      

        self.save_dir = osp.join(self.trainer_cfg['save_dir'], cfgs['data_cfg']['data_dir'], 
                                 loss_dir, self.model_type, self.avg_type, self.inference_type)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        restore_hint = self.trainer_cfg['restore_hint']
        self.load(restore_hint)

    def build_model(self, model_cfg):
        Model = getattr(models, model_cfg['type'])
        valid_args = get_valid_args(Model, model_cfg, ['type'])
        model = Model(**valid_args)
        model = model.to(self.device)
        return model

    def build_loaders(self, data_cfg, loader_cfg):
        Dataset = getattr(dataset, data_cfg['type'])
        valid_dataset_args = get_valid_args(Dataset, data_cfg, ['type'])
        train_dataset = Dataset(flag='train', **valid_dataset_args)
        test_dataset = Dataset(flag='test', **valid_dataset_args)
        train_val_dataset = Dataset(flag='train_val', **valid_dataset_args)

        valid_loader_args = get_valid_args(DataLoader, loader_cfg)
        train_loader = DataLoader(train_dataset, shuffle=True, **valid_loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **valid_loader_args)
        train_loader_val = DataLoader(train_val_dataset, shuffle=False, **valid_loader_args)

        return train_loader, test_loader, train_loader_val
    
    def get_optimizer(self, optimizer_cfg):
        # 注意这里在统计参数的时候, 不要忽略在损失函数中的参数, 例如: center loss中的center参数
        Optimizer = getattr(optim, optimizer_cfg['type'])
        valid_arg = get_valid_args(Optimizer, optimizer_cfg, ['type'])
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        loss_params = list(filter(lambda p: p.requires_grad, self.criterion.parameters()))
        params = model_params + loss_params
        optimizer = Optimizer(params, **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        Scheduler = getattr(optim.lr_scheduler, scheduler_cfg['type'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['type'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler
    
    def compute_loss(self, logits, labels, feats, regions):
        if self.loss_type == "Binary":
            total_loss = 0
            for i in range(self.num_classes):
                # num_classes实际上是region的数量, 每种类型region的损失分别计算
                i_index = (regions == i).nonzero().squeeze(-1) # 获取region i的索引, shape: [num_samples]
                if i_index.numel() == 0: # 避免没有region的情况
                    continue 
                i_logits = logits[i_index] # shape: [num_samples, num_classes]
                # 这里应该根据region来获取最后一个维度的logits
                i_logits = i_logits[:, i:i+1] # shape: [num_samples, 1]  
                i_feats = feats[i_index]
                i_labels = labels[i_index]
                i_loss, _ = self.criterion(i_logits, i_feats, i_labels)
                # print(f'Loss for region {i}: {i_loss:.6f}, num_samples: {i_index.numel()}')
                total_loss += i_loss
            loss = total_loss / self.num_classes
        else:
            print(logits.shape)
            loss, loss_info = self.criterion(logits, feats, labels)
            print(f'Loss: {loss:.6f}, Loss Info: {loss_info}')
        return loss
    
    def train(self):
        self.model.train()
        train_loss_list = []
        test_loss_list = []
        no_improvement = 0
        for i in range(self.start_epoch+1, self.epochs+1):
            print(f'############ Epoch: {i} start ###############')
            train_loss = []
            preds = []
            probs = []
            gts = []
            for batch_idx, (imgs, labels, regions) in enumerate(self.train_loader):
                imgs, labels, regions = imgs.to(self.device), labels.to(self.device), regions.to(self.device)
                self.optimizer.zero_grad()
                logits, feats = self.model(imgs)
                loss = self.compute_loss(logits=logits, labels=labels, feats=feats, regions=regions)
                pred, prob = self.logits_inference(logits, feats, regions)
                preds.append(ts2np(pred))
                probs.append(ts2np(prob))
                gts.append(ts2np(labels))
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            self.lr_scheduler.step() # 调整学习率

            train_loss = np.average(train_loss)
            preds, gts = np.concatenate(preds), np.concatenate(gts)
            probs = np.concatenate(probs)
            train_metrics = cal_metrics(gts, preds, probs, avg_type=self.avg_type, stage='Training')
            test_loss, test_metrics = self.test(model_epoch=i, is_testing=True) # 在训练阶段进行测试
            test_average = test_metrics['average']
            
            if test_average > self.best_test_metric:
                self.best_test_metric = test_average
                self.save(i, test_average, test_best=True) # 保存最好的测试模型
                no_improvement = 0
            else:
                no_improvement += 1
                if test_average == self.best_test_metric:
                    self.save(i, test_average, test_best=True)

            print(f'Epoch: {i}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            for key, value in train_metrics.items(): # 打印训练集和测试集的指标
                print(f'Training {key}: {value:.4f}, Test {key}: {test_metrics[key]:.4f}')

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            print() # 换行
            
            if no_improvement >= self.early_stop:
                print(f'Early stopping at epoch {i}')
                break
        
        # 保存训练过程中的loss曲线
        plot_loss(train_loss_list, test_loss_list, self.save_dir)

        # 加载表现最好的模型, 并测试
        self.load(epoch=0)
        self.test(is_testing=True)


    def test(self, model_epoch=None, is_testing=True):
        self.model.eval()
        total_loss = []
        preds = []
        probs = []
        gts = []
        feats_list = []
        with torch.no_grad():
            for data, target, region in self.test_loader:
                data, target, region = data.to(self.device), target.to(self.device), region.to(self.device)
                logits, feats = self.model(data)
                loss = self.compute_loss(logits=logits, labels=target, feats=feats, regions=region)
                pred, prob = self.get_preds(logits, feats, region)
                feats_list.append(ts2np(feats))
                preds.append(ts2np(pred))
                probs.append(ts2np(prob))
                gts.append(ts2np(target))
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        probs = np.concatenate(probs)
        metrics = cal_metrics(gts, preds, probs, avg_type=self.avg_type, stage='Test')

        # 用于可视化的阈值
        vis_threshold = self.best_test_metric

        # 对于最佳模型或者在测试阶段, 可视化混淆矩阵和特征
        if metrics['average'] >= vis_threshold or is_testing:
            model_epoch = self.start_epoch if model_epoch is None else model_epoch
            for key, value in metrics.items():
                print(f'Test {key} on epoch {model_epoch}: {value:.4f}')
            confusion_matrix = get_confusion_matrix(gts, preds)
            plot_confusion_matrix(confusion_matrix, self.save_dir, save_name=f'epoch_{model_epoch}_test_confusion_matrix.jpg')
            # 可视化特征
            feats_list = np.concatenate(feats_list, axis=0)
            plot_features(feats_list, gts, self.save_dir, save_name=f'epoch_{model_epoch}_test_features.jpg')
        
        if is_testing and self.inference_type == "metric":
            self.save_protos() # test完毕后保存prototypes和thresholds
        
        self.model.train()
        return total_loss, metrics

    def get_preds(self, logits, feats, regions):
        if self.inference_type == "metric":
            preds, probs = self.metric_inference(logits, feats, regions)
        else:
            preds, probs = self.logits_inference(logits, feats, regions)
        return preds, probs
    
    def save_protos(self):
        proto_path = os.path.join(self.save_dir, 'protos.pt')
        threshold_path = os.path.join(self.save_dir, 'thresholds.pt')
        torch.save(self.region_prototypes, proto_path)
        torch.save(self.region_thresholds, threshold_path)
        print("Prototypes and Thresholds saved")
    
    def load_protos(self):
        proto_path = os.path.join(self.save_dir, 'protos.pt')
        threshold_path = os.path.join(self.save_dir, 'thresholds.pt')
        
        if os.path.exists(proto_path) and os.path.exists(threshold_path):
            print("Loading Prototypes and Thresholds")
            self.region_prototypes = torch.load(proto_path)
            self.region_thresholds = torch.load(threshold_path)
            return True
        else:
            return False
    
    def get_prototypes(self):
        if self.loss_type == "Binary":
            n_regions = self.model.num_classes
        else:
            n_regions = self.model.num_classes // 2
        real_feats_list = [[] for _ in range(n_regions)]
        # iterate over the trainloader to get the prototypes
        with torch.no_grad():
            for data, target, region in self.train_loader_val: # avoid shuffling
                data, target, region = data.to(self.device), target.to(self.device), region.to(self.device)
                logits, feats = self.model(data)
                # get the real features
                if self.loss_type == "Binary":
                    real_index = (target == 1).nonzero().squeeze(-1)
                else:
                    real_index = (target >= n_regions).nonzero().squeeze(-1)
                real_feats = feats[real_index]
                real_regions = region[real_index]
                for i in range(n_regions):
                    i_index = (real_regions == i).nonzero().squeeze(-1)
                    i_feats = real_feats[i_index]
                    real_feats_list[i].append(i_feats)
        # get the average features
        region_prototypes = []
        region_thresholds = []
        for feats_list in real_feats_list:
            feats = torch.cat(feats_list, dim=0)
            mean_feats = torch.mean(feats, dim=0)
            # compute the dist
            dist = torch.norm(feats - mean_feats, dim=1)
            max_val = torch.max(dist)
            
            region_prototypes.append(mean_feats)
            region_thresholds.append(max_val)
            
        return torch.vstack(region_prototypes), torch.vstack(region_thresholds)
    
    
    def metric_inference(self, logits, feats, regions):
        if not self.load_protos():
            self.region_prototypes, self.region_thresholds = self.get_prototypes()
        num_regions = self.region_prototypes.size(0)
        # get the distance between the features and the prototypes
        compared_feats = self.region_prototypes[regions] # get the prototype of the region, shape: [batch_size, feat_dim]
        distance = torch.norm(feats - compared_feats, dim=1) # shape: [batch_size]
        if self.trainer_cfg['metric_threshold'] > 0:
            thresholds = self.trainer_cfg['metric_threshold'] # shape: [batch_size]
        else:
            thresholds = self.region_thresholds[regions].squeeze(-1) * self.alpha # shape: [batch_size]
        
        preds = distance <= thresholds
        
       
        scaling_factor = 10.0 # the scaling factor for the sigmoid function
        probs = torch.sigmoid(-scaling_factor * (distance - thresholds)) # shape: [batch_size]
        fake_probs = torch.ones_like(probs) - probs # shape: [batch_size]
        
        confidence = preds.float() * probs + (1 - preds.float()) * fake_probs
        
        num_class = self.model.num_classes
        
        other_probs_per_class = (1 - confidence) / (num_class - 1) # shape: [batch_size]
        
        combined_probs = other_probs_per_class.unsqueeze(-1).repeat(1, num_class) # shape: [batch_size, num_class]
        
        # convert the binary prediction to the num_classes prediction
        if self.loss_type == "Binary":
            preds = preds.int()
        else:
            preds = preds.int()*num_regions + regions # shape: [batch_size]
        
        # set the probs of the predicted region to the confidence
        combined_probs[torch.arange(combined_probs.size(0)), preds] = confidence
        
        return preds, combined_probs
            
    def logits_inference(self, logits, feats, regions):
        if self.loss_type == "Binary":
            # get the pred logits for each region
            region_logits = logits[torch.arange(logits.size(0)), regions].unsqueeze(1) # shape: [batch_size, 1]
            preds = (region_logits > 0).int()
            probs = torch.sigmoid(region_logits) # shape: [batch_size, 1]
        else:
            preds = logits.argmax(dim=1, keepdim=True)
            probs = torch.softmax(logits, dim=1) # shape: [batch_size, num_classes]
        return preds, probs

    
    def save(self, epoch, test_average, test_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'test_average': test_average,
        }
        if test_best:  
            print(f'Saving model at epoch {epoch} with test average metric: {test_average:.4f} on test set')
            torch.save(ckpt, f'{self.save_dir}/best_test_model.pt')
        torch.save(ckpt, f'{self.save_dir}/model_epoch_{epoch}.pt')
        

    def load(self, epoch=0):
        if epoch == 0:
            path = os.path.join(self.save_dir, 'best_test_model.pt')
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']
            self.best_test_metric = ckpt['test_average']
            print(f"Resuming checkponit from epoch: {self.start_epoch}")
        else:
            print(f'No checkpoint found at {path}, starting from scratch')
    

