import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set the GPU to use
import trainers as trainers
import argparse
from utils.common import config_loader


parser = argparse.ArgumentParser(description='Main program for cigarette recognition.')
parser.add_argument('--cfgs', type=str,
                    default='configs/base.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train / test ")
parser.add_argument('--iter', type=int, default=0, help="iter to restore")
opt = parser.parse_args()
  
def main(cfgs, phase):
    Trainer = getattr(trainers, cfgs['trainer_cfg']['type']) # 获取训练器类别
    trainer = Trainer(cfgs) # 实例化训练器
    if phase == 'train':
        trainer.train()
    elif phase == 'test':
        trainer.test()

if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    if opt.iter > 0:
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
    main(cfgs, opt.phase)