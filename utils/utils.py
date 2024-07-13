from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE

def plot_features(feats, labels, save_dir, save_name='features.jpg', save_features = True):
    """
        对t-SNE降维后的特征进行可视化

    Args:
        feats (np.ndarray): shape: [num_samples, num_features]
        labels (np.ndarray): shape: [num_samples]
    """
    plt.clf()
    # t-SNE降维
    feats_embedded = TSNE(n_components=2).fit_transform(feats)
    class_names = [f'class_{i}' for i in np.unique(labels)]
    if save_features:
        # 保存降维后的特征和标签
        file_name = save_name.split('.')[0]
        feats_name = file_name + '_feats.npy'
        labels_name = file_name + '_labels.npy'
        np.save(osp.join(save_dir, feats_name), feats_embedded)
        np.save(osp.join(save_dir, labels_name), labels)
    # 绘制散点图
    scatter = plt.scatter(feats_embedded[:, 0], feats_embedded[:, 1], c=labels)
    # 添加图例
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.colorbar()
    plt.savefig(osp.join(save_dir, save_name))

def calculate_specificity(y_true, y_pred):
    """
    计算二分类和多分类问题的特异度。

    参数：
    y_true (array-like): 真实标签
    y_pred (array-like): 预测标签

    返回：
    specificity (float): 特异度
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificity_per_class = []

    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # 真负
        fp = np.sum(cm[:, i]) - cm[i, i]  # 假正
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    return np.mean(specificity_per_class)

def cal_metrics(gts, preds, probs, avg_type="binary" , stage='Test'):
    """
        计算分类指标
    Args:
        gts (np.ndarray): shape: [num_samples]
        preds (np.ndarray): shape: [num_samples]
        probs (np.ndarray): shape: [num_samples] or [num_samples, num_classes]
        avg_type (str, optional): 用于指示计算多分类指标的方式. Defaults to "binary".
        stage (str, optional): 用于指示阶段. Defaults to 'Test'.
    Returns:
        dict: 包含accuracy, precision, recall, f1, auc的字典
    """
    assert stage in ['Training', 'Test'], f'stage should be one of [Training, Test], but got {stage}'
    accuracy = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average=avg_type, zero_division=np.nan) # add zero_division to avoid warning
    recall = recall_score(gts, preds, average=avg_type, zero_division=np.nan)
    specificity = calculate_specificity(gts, preds)
    f1 = f1_score(gts, preds, average=avg_type, zero_division=np.nan)
    if avg_type == 'binary':
        auc = roc_auc_score(gts, probs, average='macro', multi_class='ovr')
    else:
        num_classes = probs.shape[1]
        gts = np.eye(num_classes)[gts]
        auc = roc_auc_score(gts, probs, average='macro', multi_class='ovr')
    average = (accuracy + precision + recall + f1 + auc) / 5
    dict_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1, 'auc': auc, 'average': average}
    return dict_metrics

def get_confusion_matrix(trues, preds):
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix

def plot_confusion_matrix(conf_matrix, save_dir, save_name='confusion_matrix.jpg'):
    plt.clf()
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = []
    for i in range(len(conf_matrix)):
        labels.append(i)

    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    # plot data
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig(osp.join(save_dir, save_name))


def plot_loss(train_loss_list, test_loss_list, save_dir):
    plt.clf()
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.legend()
    plt.savefig(osp.join(save_dir, 'loss.jpg'))

