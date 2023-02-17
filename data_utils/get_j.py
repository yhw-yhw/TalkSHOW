import torch


def to3d(poses, config):
    if config.Data.pose.convert_to_6d:
        if config.Data.pose.expression:
            poses_exp = poses[:, -100:]
            poses = poses[:, :-100]

        poses = poses.reshape(poses.shape[0], -1, 5)
        sin, cos = poses[:, :, 3], poses[:, :, 4]
        pose_angle = torch.atan2(sin, cos)
        poses = (poses[:, :, :3] * pose_angle.unsqueeze(dim=-1)).reshape(poses.shape[0], -1)

        if config.Data.pose.expression:
            poses = torch.cat([poses, poses_exp], dim=-1)
    return poses


def get_joint(smplx_model, betas, pred):
    joint = smplx_model(betas=betas.repeat(pred.shape[0], 1),
                        expression=pred[:, 165:265],
                        jaw_pose=pred[:, 0:3],
                        leye_pose=pred[:, 3:6],
                        reye_pose=pred[:, 6:9],
                        global_orient=pred[:, 9:12],
                        body_pose=pred[:, 12:75],
                        left_hand_pose=pred[:, 75:120],
                        right_hand_pose=pred[:, 120:165],
                        return_verts=True)['joints']
    return joint


def get_joints(smplx_model, betas, pred):
    if len(pred.shape) == 3:
        B = pred.shape[0]
        x = 4 if B>= 4 else B
        T = pred.shape[1]
        pred = pred.reshape(-1, 265)
        smplx_model.batch_size = L = T * x

        times = pred.shape[0] // smplx_model.batch_size
        joints = []
        for i in range(times):
            joints.append(get_joint(smplx_model, betas, pred[i*L:(i+1)*L]))
        joints = torch.cat(joints, dim=0)
        joints = joints.reshape(B, T, -1, 3)
    else:
        smplx_model.batch_size = pred.shape[0]
        joints = get_joint(smplx_model, betas, pred)
    return joints