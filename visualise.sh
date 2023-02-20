python -W ignore scripts/diversity.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_pixel.json \
--face_model_path ./experiments/2022-10-15-smplx_S2G-face-3d/ckpt-99.pth \
--body_model_path ./experiments/2022-11-02-smplx_S2G-body-pixel-3d/ckpt-99.pth \
--infer