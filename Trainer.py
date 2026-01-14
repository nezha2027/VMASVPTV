import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from utils import MetricLogger, fakecast
from utils import show_pcd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(1)
import re
from savelog import save_logs

class Trainer:
    """
    训练器，输入待训练的模型、参数，封装训练过程
    """
    def __init__(self, args, model, optimizer, scheduler, criterion, dataset, mode, scratch=False):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.dataset = dataset
        self.mode = mode
        self.dataloader = None
        self.epoch = 1
        self.step = 1
        self.scratch = scratch


        self.epoch_metric_logger = MetricLogger()

        # 恢复检查点
        if self.args.checkpoint != '':
            checkpoint = torch.load(self.args.checkpoint, map_location=self.args.device)
            # Load model
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                if model_state_dict.keys() != self.model.state_dict().keys():
                    logger.info("Load model Failed, keys not match..")
                else:
                    self.model.load_state_dict(model_state_dict)
                    logger.info("Load model state")
                    if 'optimizer' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        logger.info("Load optimizer state")
                    if 'scheduler' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler'])
                        logger.info("Load scheduler state")

            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch'] + 1
                logger.info(f"Load epoch, current = {self.epoch}")

            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1
                logger.info(f"Load step, current = {self.step}")
            logger.info(f'Load checkpoint complete: \'{self.args.checkpoint}\'')
        else:
            logger.info(f'{mode} with an initial model')
            
        #检查训练器的配置参数 self.args 中是否存在一个名为 auto_cast 的属性。这个属性是一个布尔值，用于指示是否启用自动混合精度。
        if self.args.auto_cast:
            self.cast = autocast
        else:
            self.cast = fakecast 
            

        # 创建训练、测试结果保存目录
        if self.scratch:
            self.log = f'{self.args.name}_model={self.args.model_cfg}_aug={self.args.data_aug}_' \
                f'lr={self.args.lr}_wd={self.args.wd}_bs={self.args.batch_size}_' \
                f'{self.args.optimizer}_{self.args.scheduler}_'\
                    f'160_cr={args.criterion}_scratch'
        else:
            self.log = f'{self.args.name}_model={self.args.model_cfg}_aug={self.args.data_aug}_' \
                    f'lr={self.args.lr}_wd={self.args.wd}_bs={self.args.batch_size}_' \
                    f'{self.args.optimizer}_{self.args.scheduler}_'\
                        f'160_cr={args.criterion}'
        print('log: ', self.log)
        if self.mode == 'train':
            self.save_root = os.path.join('./result_train', self.log)
        elif self.mode == 'test':
            self.save_root = os.path.join('./result_test', self.log)
        else:
            raise ValueError
        os.makedirs(self.save_root, exist_ok=True)
        logger.info(f'save root = \'{self.save_root}\'')
        logger.info(f'run in {self.args.device}')

    def run(self):
        if self.mode == 'train':
            self.train()
            test_metric = self.test()  #验证
        elif self.mode == 'test':
            self.test()
        return test_metric

    def train(self):
        # tensorboard可视化训练过程，记录训练时的相关数据，使用指令:tensorboard --logdir=runs
        self.writer = SummaryWriter(os.path.join('./runs', self.log))

        start_epoch = self.epoch
        # init eval
        # self.test(init=True)
        for ep in range(start_epoch, self.args.num_epochs + 1):
            # 记录日志
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], ep)

            # 单轮训练
            self.train_one_epoch()

            # 动态学习率
            self.scheduler.step()

            # 定期保存
            if self.epoch % self.args.save_cycle == 0:
                self.save()

            # 定期验证
            if self.epoch % self.args.eval_cycle == 0:
                self.test()

            self.epoch += 1

        self.save(finish=True)

    def train_one_epoch(self):
        self.model.train()
        self.dataset.train()
        # save_logs(self.args.log_save_file)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers,# self.args.num_workers
                                     shuffle=True,# False
                                     pin_memory=True,
                                     drop_last=False) 
        # froze BN
        if self.args.train_bn is False:
            import torch.nn as nn
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    module.eval()


        epoch_loss, epoch_acc = [], []
        count = self.args.log_cycle // self.args.batch_size

        loop = tqdm(self.dataloader, total=len(self.dataloader), leave=False) #创建一个 tqdm 进度条对象 loop
        loop.set_description('train'+str(self.epoch)) #设置进度条的描述
        for data in loop: #循环从数据加载器中获取每个批次的数据。
            pcd, label, condi_d = data #解包每个批次的数据 #!!!
            # show_pcd([pcd[0].T], normal=True)
            pcd, label, condi_d = pcd.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True),condi_d.to(self.args.device, non_blocking=True)

            # 前向传播与反向传播
            with self.cast():
                # points_cls = self.model(pcd) 
                points_cls = self.model(pcd, condi_d) #!!!
                points_cls = points_cls.squeeze(1)
                loss = self.criterion(points_cls, label, pcd)
            # print(loss)
            self.epoch_metric_logger.add_metric('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录日志
            # loop.set_postfix(train_loss=loss.item(), acc=f'{acc * 100:.2f}%')
            epoch_loss.append(loss.item())
            # epoch_acc.append(acc)
            count -= 1
            if count <= 0:
                count = self.args.log_cycle // self.args.batch_size
                self.writer.add_scalar("train/step_loss", sum(epoch_loss[-count:]) / count, self.step)
                # print('train/step_loss', sum(epoch_loss[-count:]) / count, 'in epoch', self.epoch)
                # self.writer.add_scalar("train/step_acc", sum(epoch_acc[-count:]) / count, self.step)
                self.step += 1
        loop.set_postfix(lr=self.optimizer.param_groups[0]['lr'])
        self.writer.add_scalar("train/epoch_loss", sum(epoch_loss) / len(epoch_loss), self.epoch)
        # print('Train MSE ', sum(epoch_loss) / len(epoch_loss), 'in epoch', self.epoch)
        # print(f'Train Epoch {self.epoch:>4d} ' + self.epoch_metric_logger.tostring())
        # self.writer.add_scalar("train/epoch_acc", sum(epoch_acc) / len(epoch_acc), self.epoch)
        logger.info(f'Train Epoch {self.epoch:>4d} ' + self.epoch_metric_logger.tostring())
        self.epoch_metric_logger.clear()

    def test(self, init=False):
        self.model.eval()   #!!!
        self.dataset.eval()
        if self.mode == 'eval':
            self.epoch -= 1
        # self.dataset.transforms.set_padding(False)
        eval_dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=16,
                                     num_workers=min(self.args.num_workers, 16),
                                     pin_memory=True,
                                     drop_last=False,
                                     shuffle=False)
        
        loop = tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)
        loop.set_description('eval')
        # torch.manual_seed(0)# ! seed fixed
        loss_tmp = []
        for data in loop: 
            pcd, label, condi_d = data
            # show_pcd([pcd[0].T], normal=True)
            pcd, label, condi_d = pcd.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True), condi_d.to(self.args.device, non_blocking=True)

            # 前向传播
            with torch.no_grad():
                # pcd, _, _ = self.uniform_pc(pcd)
                # print(torch.max(pcd,axis=2))
                # points_cls = self.model(pcd)
                points_cls = self.model(pcd, condi_d)
                points_cls = points_cls.squeeze(1)
                loss = self.criterion(points_cls, label, pcd)
                print(loss)
                loss_tmp.append(loss.item())

            self.epoch_metric_logger.add_metric('loss', loss.item())
            # self.epoch_metric_logger.add_metric('acc', acc)

            loop.set_postfix(eval_loss=loss.item())
        # export loss
        # n = len(loss_tmp)//4
        # avg_loss = []
        # for i in range(4):
        #     start = i * n
        #     end = (i+1) * n
        #     sub_loss = loss_tmp[start:end]
        #     avg_loss_tmp = sum(sub_loss)/n
        #     avg_loss.append(avg_loss_tmp)

        # self.dataset.transforms.set_padding(True)
        # import numpy as np
        # print('np meanloss ',np.mean(self.epoch_metric_logger.metrics['loss']), 'len', len(self.epoch_metric_logger.metrics['loss']))
        if init:
            print('Eval MSE ', self.epoch_metric_logger.tostring(), 'initial')
        else:
            print('Eval MSE ', self.epoch_metric_logger.tostring(), 'in epoch', self.epoch)
        metric = self.epoch_metric_logger.get_average_value()

        if self.mode == 'train':
            self.writer.add_scalar("eval/loss", metric['loss'], self.epoch)
            # self.writer.add_scalar("eval/acc", metric['acc'], self.epoch)
        self.epoch_metric_logger.clear()
        return loss_tmp

    def save(self, finish=False):
        model_state_dict = self.model.state_dict()
        if not finish:
            state = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}_epoch{self.epoch}.pth')
        else:
            state = {
                'model': model_state_dict,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}.pth')

        torch.save(state, file_path)


