import numpy as np
import torch

lower_pose = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0747, -0.0158, -0.0152, -1.1826512813568115, 0.23866955935955048,
     0.15146760642528534, -1.2604516744613647, -0.3160211145877838,
     -0.1603458970785141, 1.1654603481292725, 0.0, 0.0, 1.2521806955337524, 0.041598282754421234, -0.06312154978513718,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
lower_pose_stand = torch.tensor([
    8.9759e-04, 7.1074e-04, -5.9163e-06, 8.9759e-04, 7.1074e-04, -5.9163e-06,
    3.0747, -0.0158, -0.0152,
    -3.6665e-01, -8.8455e-03, 1.6113e-01, -3.6665e-01, -8.8455e-03, 1.6113e-01,
    -3.9716e-01, -4.0229e-02, -1.2637e-01,
    7.9163e-01, 6.8519e-02, -1.5091e-01, 7.9163e-01, 6.8519e-02, -1.5091e-01,
    7.8632e-01, -4.3810e-02, 1.4375e-02,
    -1.0675e-01, 1.2635e-01, 1.6711e-02, -1.0675e-01, 1.2635e-01, 1.6711e-02, ])
# lower_pose_stand = torch.tensor(
#     [6.4919e-02,  3.3018e-02,  1.7485e-02,  8.9759e-04,  7.1074e-04, -5.9163e-06,
#      3.0747, -0.0158, -0.0152,
#      -3.3633e+00, -9.3915e-02, 3.0996e-01, -3.6665e-01, -8.8455e-03, 1.6113e-01,
#      1.1654603481292725, 0.0, 0.0,
#      4.4167e-01,  6.7183e-03, -3.6379e-03,  7.9163e-01,  6.8519e-02, -1.5091e-01,
#      0.0, 0.0, 0.0,
#      2.2910e-02, -2.4797e-02, -5.5657e-03, -1.0675e-01,  1.2635e-01,  1.6711e-02,])
lower_body = [0, 1, 3, 4, 6, 7, 9, 10]
count_part = [6, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 36, 37,
              38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
fix_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
all_index = np.ones(275)
all_index[fix_index] = 0
c_index = []
i = 0
for num in all_index:
    if num == 1:
        c_index.append(i)
    i = i + 1
c_index = np.asarray(c_index)

fix_index_3d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                21, 22, 23, 24, 25, 26,
                30, 31, 32, 33, 34, 35,
                45, 46, 47, 48, 49, 50]
all_index_3d = np.ones(165)
all_index_3d[fix_index_3d] = 0
c_index_3d = []
i = 0
for num in all_index_3d:
    if num == 1:
        c_index_3d.append(i)
    i = i + 1
c_index_3d = np.asarray(c_index_3d)

c_index_6d = []
i = 0
for num in all_index_3d:
    if num == 1:
        c_index_6d.append(2*i)
        c_index_6d.append(2 * i + 1)
    i = i + 1
c_index_6d = np.asarray(c_index_6d)


def part2full(input, stand=False):
    if stand:
        # lp = lower_pose_stand.unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)
        lp = torch.zeros_like(lower_pose)
        lp[6:9] = torch.tensor([3.0747, -0.0158, -0.0152])
        lp = lp.unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)
    else:
        lp = lower_pose.unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)

    input = torch.cat([input[:, :3],
                       lp[:, :15],
                       input[:, 3:6],
                       lp[:, 15:21],
                       input[:, 6:9],
                       lp[:, 21:27],
                       input[:, 9:12],
                       lp[:, 27:],
                       input[:, 12:]]
                      , dim=1)
    return input


def pred2poses(input, gt):
    input = torch.cat([input[:, :3],
                       gt[0:1, 3:18].repeat(input.shape[0], 1),
                       input[:, 3:6],
                       gt[0:1, 21:27].repeat(input.shape[0], 1),
                       input[:, 6:9],
                       gt[0:1, 30:36].repeat(input.shape[0], 1),
                       input[:, 9:12],
                       gt[0:1, 39:45].repeat(input.shape[0], 1),
                       input[:, 12:]]
                      , dim=1)
    return input


def poses2poses(input, gt):
    input = torch.cat([input[:, :3],
                       gt[0:1, 3:18].repeat(input.shape[0], 1),
                       input[:, 18:21],
                       gt[0:1, 21:27].repeat(input.shape[0], 1),
                       input[:, 27:30],
                       gt[0:1, 30:36].repeat(input.shape[0], 1),
                       input[:, 36:39],
                       gt[0:1, 39:45].repeat(input.shape[0], 1),
                       input[:, 45:]]
                      , dim=1)
    return input

def poses2pred(input, stand=False):
    if stand:
        lp = lower_pose_stand.unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)
        # lp = torch.zeros_like(lower_pose).unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)
    else:
        lp = lower_pose.unsqueeze(dim=0).repeat(input.shape[0], 1).to(input.device)
    input = torch.cat([input[:, :3],
                       lp[:, :15],
                       input[:, 18:21],
                       lp[:, 15:21],
                       input[:, 27:30],
                       lp[:, 21:27],
                       input[:, 36:39],
                       lp[:, 27:],
                       input[:, 45:]]
                      , dim=1)
    return input


rearrange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]\
            # ,22, 23, 24, 25, 40, 26, 41,
            #  27, 42, 28, 43, 29, 44, 30, 45, 31, 46, 32, 47, 33, 48, 34, 49, 35, 50, 36, 51, 37, 52, 38, 53, 39, 54, 55,
            #  57, 56, 59, 58, 60, 63, 61, 64, 62, 65, 66, 71, 67, 72, 68, 73, 69, 74, 70, 75]

symmetry = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]#, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # 1, 1, 1, 1, 1, 1]
