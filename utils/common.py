import yaml
import torch
import inspect
import logging
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v

def get_valid_args(obj, input_args, free_keys=[]):
    """
       获取对象的有效参数, 即对象的__init__方法中的参数, 或者函数的参数
        :param obj: 对象
        :param input_args: 输入参数
        :param free_keys: 自由参数, 即不需要检查的参数
    """
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def MergeCfgsDict(src, dst):
    """
        合并两个配置字典
    """
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v

def get_base_config():
    with open("./configs/base.yaml", 'r', encoding='utf-8') as stream:
        dst_cfgs = yaml.safe_load(stream)
    return dst_cfgs

def config_loader(path):
    """
        加载配置文件
    """
    with open(path, 'r', encoding='utf-8') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open("./configs/base.yaml", 'r', encoding='utf-8') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)

def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)

def is_ndarray(x):
    return isinstance(x, np.ndarray)

def is_tensor(x):
    return isinstance(x, torch.Tensor)

def ts2np(x):
    return x.cpu().data.numpy()

def np2ts(x):
    return torch.from_numpy(x).cuda()

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)