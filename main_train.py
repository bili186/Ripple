
import os
import os.path as osp
import time
import yaml
import torch

import argparse
import torch.distributed as dist

from glob import glob
from opensphere.utils.build_helper import build_from_args

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def get_config_from_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch framework for deep metric learning.')
    parser.add_argument('--cfg_path', default='',
            help='path of training config')
    parser.add_argument('--proj_dir', default='./project',
            help='the dir to save logs and models')
    parser.add_argument('--start_time', default='20220927_160047',
            help='time to start training')
    args = parser.parse_args()

    ckpt_paths = glob(osp.join(args.proj_dir, 'checkpoint/ckpt_*.pth'))
    if len(ckpt_paths) > 0:
        cfg_path = osp.join(args.proj_dir, 'config.yml')
    else:
        cfg_path = args.cfg_path

    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.SafeLoader)

    trainer_cfg = config['trainer']
    trainer_cfg['proj_dir'] = args.proj_dir

    trainer_cfg['device_ids'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    if trainer_cfg['device_ids'] is None:
        raise KeyError('Please specify GPU IDs.')

    start_time = time.strptime(args.start_time, '%Y%m%d_%H%M%S')
    while time.localtime() < start_time:
        print(args.start_time)
        time.sleep(666)
    print('start...')

    return config

def main_worker(config):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size,
    )

    trainer_args = {'type': config['trainer']['type'], 'config': config}
    trainer = build_from_args(trainer_args, 'opensphere')
    trainer.run()

    dist.destroy_process_group()

if __name__ == '__main__':
    config = get_config_from_args()
    main_worker(config)