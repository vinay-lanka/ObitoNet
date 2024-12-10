import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN


class PCTokenizer (nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
    
    def forward(self, xyz):
        neighborhood, center = self.knn_gather(xyz)
        

    def knn_gather(self, xyz):
            '''
                input: B N 3
                ---------------------------
                output: B G M 3
                center : B G 3
            '''
            batch_size, num_points, _ = xyz.shape
            # fps the centers out
            center = self.fps(xyz, self.num_group) # B G 3
            # knn to get the neighborhood
            _, idx = self.knn(xyz, center) # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            # normalize
            neighborhood = neighborhood - center.unsqueeze(2)
            return neighborhood, center
    
    def fps(self, data, number):
        '''
            data B N 3
            number int
        '''
        # print(number)
        # print("yoyoyo",data.scalar_type())
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_data
    
class ObitoNet (nn.module):
    def __init__(self):
        self.group_size = 64
        self.num_groups = 32
        self.pc_tokenizer = PCTokenizer(self.group_size, self.num_groups)
    def forward(self, pts):
        tokens = self.pc_tokenizer(pts)