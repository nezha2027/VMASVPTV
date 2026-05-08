import argparse
import sys
import os


def str_to_bool(s):
    if (s.lower() == 'true'):
        return True
    elif (s.lower() == 'false'):
        return False
    else:
        raise TypeError(f'str {s} can not convert to bool.')

parser = argparse.ArgumentParser(description='Feature Extractor for Alignment')

parser.add_argument('--name',                   default='PointNeXt',            type=str,
                    help='Name of the model')
parser.add_argument('--mode',                   default='train',                type=str,
                    choices=['train', 'test'],
                    help='train or test')

# parser.add_argument('--dataset',                default='waverider',           type=str,
#                     choices=['waverider','capsule','slender','spaceplane','X-33'],
#                     help='Dataset name')
# parser.add_argument('--dataset_name',           default='shapes_N5_D30_221_seed1_RTY',                   type=str,
#                     help='Name to dataset path') 
parser.add_argument('--dataset',                default='save_pcs',           type=str,
                    choices=['save_pcs'],
                    help='Dataset name')
parser.add_argument('--dataset_name',                default='160conditions_Ma3',           type=str,
                    help='Name to dataset path')

parser.add_argument('--checkpoint_epoch',           default='300', type=str,
                    help='epoch of souce checkpoint')
parser.add_argument('--cross_val', '-cv',      default=1,                     type=int,
                    help='cv')


parser.add_argument('--criterion', '-cr',      default='mse',                     type=str,
                    help='loss function')
parser.add_argument('--batch_size', '-bs',      default=16,                     type=int,
                    help='Batch size for training') #!!!
parser.add_argument('--num_epochs', '-ne',      default=300,                    type=int,
                    help='Number of epochs for training')
parser.add_argument('--lr', '--learning-rate',  default=1e-3,                   type=float,
                    help='initial learning rate for optimizer')
parser.add_argument('--wd', '--weight_decay',   default=1e-4,                   type=float,
                    help='Weight decay for optimizer')
parser.add_argument('--momentum',               default=0.9,                    type=float,
                    help='Momentum value for optimizer')
parser.add_argument('--data_aug',               default='basic',                type=str,
                    help='configuration for data augment')
parser.add_argument('--optimizer',              default='AdamW',                type=str,
                    help='optimizer')
parser.add_argument('--scheduler',              default='cosine',               type=str,
                    help='scheduler')
parser.add_argument('--froze',              default='need_froze_list1_91.csv',               type=str,
                    help='model froze list')
parser.add_argument('-train_bn',              default='False',               type=str_to_bool,
                    help='train BatchNorm?')
parser.add_argument('--eval_num',              default=20,               type=int,
                    help='number of sample in test set') 


parser.add_argument('--model_cfg',              default='basic_c',              type=str,
                    help='Configuration for building pcd backbone')


parser.add_argument('--eval_cycle', '-ec',      default=20,                     type=int,
                    help='Evaluate every n epochs')#!!!
parser.add_argument('--log_cycle', '-lc',       default=320,                    type=int,
                    help='Log every n steps')
parser.add_argument('--save_cycle', '-sc',      default=50,                      type=int,
                    help='Save every n epochs')
parser.add_argument('--checkpoint', '-cp',      default='',                     type=str,
                    help='Checkpoint file name')

parser.add_argument('--num_workers', '-nw',     default=2,                     type=int,
                    help='Number of workers used in dataloader')
parser.add_argument('--use_cuda',               default='True',                 type=str_to_bool,
                    help='Using cuda to run')
parser.add_argument('--auto_cast',              default='False',                type=str_to_bool,
                    help='Using torch.cuda.amp.autocast to accelerate computing')
parser.add_argument('--gpu_index',              default='0',                    type=str,
                    help='Index of gpu')


aug_None = {}
aug_basic = {'RT': [0, 0, 0, 0, 1], 'jitter': [0, 0, 1], 'random_drop': 0.5, 'random_sample': True}
aug_basic_pi = {'RT': [0, 3.1415, 0, 0.1, 1], 'jitter': [0, 0.01, 1], 'random_drop': 0.5, 'random_sample': True}
DATA_AUG_CONFIG = {'None': aug_None, 'basic': aug_basic, 'basic_pi': aug_basic_pi}

basic_c = {
    'type': 'regression',
    'num_class': 40,
    'max_input': 4096,  
    'npoint': [512, 128, 32, 8],
    'radius_list': [[0.1, 0.2], [0.2, 0.4, 0.4], [0.4, 0.8], [0.8, 1.6]],
    'nsample_list': [[16, 16], [16, 16, 16], [16, 16], [8, 8]],
    'coor_dim': 3, 
    'width': 32,
    'expansion': 4,
    'normal': True,
    'head': [512, 256]
}

MODEL_CONFIG = {
    'basic_c': basic_c,
}


parser.add_argument('--conditions',              default='condi_v',              type=str,
                    help='Variable conditions')
condi_v = {
    '3':[3, 2758.44, 160.71],
    '4':[4, 1480.91, 112.857],
    '5':[5, 829.87, 77.78],
    '6':[6, 312.397, 56.5854],
    '7':[7, 187.65, 44.75],
    '8':[8, 80.9251, 35.5072],
}

CONDITION_VERY = {
    'condi_v': condi_v,
}
