from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', default='smplx_S2G', type=str)
    parser.add_argument('--speakers', nargs='+')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model_name', type=str)
    
    #for Tmpt and S2G
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--template_length', default=0, type=int)

    #for training from a ckpt
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_pth', default=None, type=str)
    parser.add_argument('--style_layer_norm', action='store_true')
    
    #required
    parser.add_argument('--config_file', default='./config/style_gestures.json', type=str)

    # for visualization and test
    parser.add_argument('--audio_file', default=None, type=str)
    parser.add_argument('--id', default=0, type=int, help='0=oliver, 1=chemistry, 2=seth, 3=conan')
    parser.add_argument('--only_face', action='store_true')
    parser.add_argument('--stand', action='store_true')
    parser.add_argument('--whole_body', action='store_true')
    parser.add_argument('--num_sample', default=1, type=int)
    parser.add_argument('--face_model_name', default='s2g_face', type=str)
    parser.add_argument('--face_model_path', default='./experiments/2022-10-15-smplx_S2G-face-3d/ckpt-99.pth', type=str)
    parser.add_argument('--body_model_name', default='s2g_body_pixel', type=str)
    parser.add_argument('--body_model_path', default='./experiments/2022-11-02-smplx_S2G-body-pixel-3d/ckpt-99.pth', type=str)
    parser.add_argument('--infer', action='store_true')

    return parser