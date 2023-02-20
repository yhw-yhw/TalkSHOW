python -W ignore scripts/test_face.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_face.json \
--face_model_name s2g_face \
--face_model_path ./experiments/2023-02-04-smplx_S2G-face/ckpt-99.pth \
--infer