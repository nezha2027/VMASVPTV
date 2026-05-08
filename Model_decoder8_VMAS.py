import torch
import torch.nn as nn
from FeatureExtractorPart.pointnext import Stage, FeaturePropagation, RegHead
from FeatureExtractorPart.utils import index_points, farthest_point_sample
# global_feature = None
def ConditionMap(in_channel, channel_list, dim=1, bias=False, drop_last_act=False,
              drop_last_norm_act=False):
    if dim == 1:
        Conv = nn.Conv1d
    else:
        Conv = nn.Conv2d
    ACT = nn.ReLU

    # 根据通道数构建mlp
    mlp = []
    for i, channel in enumerate(channel_list):
        # 每层为conv-relu
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(ACT(inplace=True))
        in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]
    elif drop_last_norm_act:
        mlp = mlp[:-2]
        mlp[-1] = Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=True)

    return nn.Sequential(*mlp)

class PointNeXt(nn.Module):
    """
    PointNeXt语义分割模型特征提取部分
    """
 
    def __init__(self, cfg):
        super().__init__()
        self.type = cfg['type'] 
        self.num_class = cfg['num_class']
        self.coor_dim = cfg['coor_dim']
        self.normal = cfg['normal']
        width = cfg['width']

        self.mlp = nn.Conv1d(in_channels=self.coor_dim + self.coor_dim * self.normal,
                             out_channels=width, kernel_size=1)
        self.stage = nn.ModuleList()

        for i in range(len(cfg['npoint'])):
            self.stage.append(
                Stage(
                    npoint=cfg['npoint'][i], radius_list=cfg['radius_list'][i], nsample_list=cfg['nsample_list'][i],
                    in_channel=width, expansion=cfg['expansion'], coor_dim=self.coor_dim
                )
            )
            width *= 2
        self.fc_mlp = nn.Conv1d(in_channels=1,
                             out_channels=8, kernel_size=1) 
        
        if self.type != 'classification':
            self.decoder = nn.ModuleList()
            for i in range(len(cfg['npoint'])):
                    in_channel = [width, width // 2, 8]
                    mlp = [width // 2, width // 2]
                    self.decoder.append(
                        FeaturePropagation(in_channel=in_channel, mlp=mlp,
                                        coor_dim=self.coor_dim)
                    )
                    width = width // 2
        # head
        self.head = RegHead(in_channel=width, mlp=cfg['head'], num_class=1, task_type=self.type)

    def forward(self, x, c):
        l0_xyz, l0_points = x[:, :self.coor_dim, :], x[:, :self.coor_dim + self.coor_dim * self.normal, :] 
  
        record = [[l0_xyz, l0_points]]
        for stage in self.stage: 
            record.append(list(stage(*record[-1])))
        self.global_feature = record[-1][1].mean(dim=2)

        fc_map = self.fc_mlp(c)
        for i, decoder in enumerate(self.decoder):
                fc_map = self.fc_mlp(c)
                fc_map = fc_map.expand(-1,-1, record[-i-2][1].shape[-1]).clone()
                record[-i-2][1] = torch.cat((record[-i-2][1], fc_map), dim=1) #!!!
                record[-i-2][1] = decoder(record[-i-2][0], record[-i-1][0], record[-i-2][1], record[-i-1][1])
        R = record[0][1] 
        points_cls = self.head(R) 
        return points_cls
    
    def get_global(self):
        return self.global_feature
