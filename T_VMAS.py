from Parameters import *
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.dirname(os.path.abspath(__name__)))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from Model_befcat_VMAS import PointNeXt
from dataset.FFDshape_VMAS import FFDshape_ptp_tl_cv
from Loss import LabelSmoothingCE, reg_loss, reg_loss_augcd, reg_loss_augcd2
from Transforms import PCDPretreatment, get_data_augment
from Trainer import Trainer
from utils import IdentityScheduler
import numpy as np
from savelog import save_logs
import re
BASEDIR = r'E:/Proceduce/VMaAOAS'

def main():
    # 解析参数
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')
    model_cfg = MODEL_CONFIG[args.model_cfg] ###!模型配置
    max_input = model_cfg['max_input']
    normal = model_cfg['normal']
    
    #给定数据集
    dataset_list = ['pc_save_train','pc_save_eval']
    dataset_path_list = []
    for dataset in dataset_list:
        dataset_path_list.append(BASEDIR+r'/PC_saveVMAS/' + dataset)
    
    #数据变换、加载数据集
    logger.info('Prepare Data')
    data_augment, random_sample, random_drop = get_data_augment(DATA_AUG_CONFIG[args.data_aug]) ###!
    transforms = PCDPretreatment(num=max_input, down_sample='random', normal=normal,
                                 data_augmentation=data_augment, random_drop=random_drop, resampling=random_sample)
    for dataset_path in dataset_path_list:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'Dataset path not found.')

    dataset = FFDshape_ptp_tl_cv(root=dataset_path_list, transforms=transforms,eval_num=args.eval_num) ###! 数据集处理
 
    # 模型与损失函数
    logger.info('Prepare Models...')
    model = PointNeXt(model_cfg).to(device=args.device) ###! 
    print('当前是训练阶段')

    if args.optimizer.lower() == 'adamw':
        Optimizer = torch.optim.AdamW
        optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        Optimizer = torch.optim.Adam
        optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'SGD':
        Optimizer = torch.optim.SGD
        optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.scheduler.lower() == 'identity':
        Scheduler = IdentityScheduler
    else:
        args.scheduler = 'cosine'
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    scheduler = Scheduler(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    if args.criterion == 'mse':
        criterion = reg_loss().to(args.device)
    elif args.criterion == 'augcd':
        criterion = reg_loss_augcd().to(args.device)
    elif args.criterion == 'augcd2':
        criterion = reg_loss_augcd2().to(args.device)

    log_name = f'{args.name}_model={args.model_cfg}_aug={args.data_aug}_' \
            f'lr={args.lr}_wd={args.wd}_bs={args.batch_size}_' \
            f'{args.optimizer}_{args.scheduler}_'\
            f'cr={args.criterion}'
    log_name = os.path.join('result_train', log_name)
    # 如果目录不存在，则创建
    log_dir = os.path.dirname(log_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_name):
        os.makedirs(log_name)

    args.log_save_file = os.path.join(log_name,'logs.txt')
    with open(args.log_save_file, 'w') as f:
        pass
    save_logs(args.log_save_file)
    
    logger.info('Trainer launching...')
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataset=dataset,   #数据集
        mode=args.mode, #train
    )
    test_metric = trainer.run()
    return test_metric


if __name__ == "__main__":
    torch.manual_seed(0)
    test_metric = main()
    test_metric = np.array(test_metric)
    np.savetxt('cache_metric.csv', test_metric, delsimiter=',')
    print('Done.')
    # with open('cache_metric.txt', 'w') as f:
    #     f.write(str(test_metric))
