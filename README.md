# TalkSHOW: Generating Holistic 3D Human Motion from Speech

The official PyTorch implementation of the paper [**"Generating Holistic 3D Human Motion from Speech"**](https://arxiv.org/abs/2212.04420).

Please visit our [**webpage**](https://talkshow.is.tue.mpg.de/) for more details.

![teaser](visualise/teaser_01.png)

## Getting started

The training code was tested on `Ubuntu 18.04.5 LTS` and the visualization code was test on `Windows 10`, and it requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)



### 1. Setup environment

Clone the repo:
  ```bash
  git clone https://github.com/yhw-yhw/TalkSHOW
  cd TalkSHOW
  ```  
Create conda environment:
```bash
conda create --name talkshow python=3.7
conda activate talkshow
```
Please install pytorch (v1.10.1).

    pip install -r requirements.txt
    
Please install [**MPI-Mesh**](https://github.com/MPI-IS/mesh).

### 2. Get data

Please note that if you only want to generate demo videos, you can skip this step and directly download the pretrained models.

Download [**SHOW_dataset_v1.0.zip**](https://download.is.tue.mpg.de/download.php?domain=talkshow&resume=1&sfile=SHOW_dataset_v1.0.zip) from [**TalkSHOW download webpage**](https://talkshow.is.tue.mpg.de/download.php),
unzip using ``for i in $(ls *.tar.gz);do tar xvf $i;done``,
and run ``python data_utils/dataset_preprocess.py`` to check and split dataset.
Modify ``data_root`` in ``config/*.json`` to the dataset-path.

### 3. Download the pretrained models (Optional)

Download [**pretrained models**](https://drive.google.com/file/d/1W0ZoUCOJ2ron5VLhRwS9c8W2_E70tIH2/view?usp=share_link),
unzip and place it in the TalkSHOW folder, i.e. ``path-to-TalkSHOW/experiments``.

### 4. Training

    # 1. Train VQ-VAEs. 
    bash train_body_vq.sh
    # 2. Train PixelCNN. Please modify "Model:vq_path" in config/body_pixel.json to the path of VQ-VAEs.
    bash train_body_pixel.sh
    # 3. Train face generator.
    bash train_face.sh

### 5. Testing

Modify the arguments in ``test_face.sh`` and ``test_body.sh``. Then

    bash test_face.sh
    bash test_body.sh

### 5. Visualization

If you ssh into the linux machine, NotImplementedError might occur. In this case, please refer to [**issue**](https://github.com/MPI-IS/mesh/issues/66) for solving the error.

Download [**smplx model**](https://drive.google.com/file/d/1Ly_hQNLQcZ89KG0Nj4jYZwccQiimSUVn/view?usp=share_link) (Please register in the official [**SMPLX webpage**](https://smpl-x.is.tue.mpg.de) before you use it.)
and place it in ``path-to-TalkSHOW/visualise/smplx_model``.
To visualise the test set and generated result (in each video, left: generated result | right: ground truth).
The videos and generated motion data are saved in ``./visualise/video/body-pixel``:
    
    bash visualise.sh

If you ssh into the linux machine, there might be an error about OffscreenRenderer. In this case, please refer to [**issue**](https://github.com/MPI-IS/mesh/issues/66) for solving the error.

To reproduce the demo videos, run
```bash
# the whole body demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/1st-page.wav --id 0 --whole_body
# the face demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 0 --only_face
# the identity-specific demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 0
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 1
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 2
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 3 --stand
# the diversity demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/style.wav --id 0 --num_samples 12
# the french demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/french.wav --id 0
# the synthetic speech demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/rich.wav --id 0
# the song demo
python scripts/demo.py --config_file ./config/body_pixel.json --infer --audio_file ./demo_audio/song.wav --id 0

````
## Citation
If you find our work useful to your research, please consider citing:
```
@article{yi2022generating,
  title={Generating Holistic 3D Human Motion from Speech},
  author={Yi, Hongwei and Liang, Hualin and Liu, Yifei and Cao, Qiong and Wen, Yandong and Bolkart, Timo and Tao, Dacheng and Black, Michael J},
  journal={arXiv preprint arXiv:2212.04420},
  year={2022}
}
```

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [Freeform](https://github.com/TheTempAccount/Co-Speech-Motion-Generation) for training pipeline
- [MPI-Mesh](https://github.com/MPI-IS/mesh), [Pyrender](https://github.com/mmatl/pyrender), [Smplx](https://github.com/vchoutas/smplx), [VOCA](https://github.com/TimoBolkart/voca) for rendering  
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) and [Faceformer](https://github.com/EvelynFan/FaceFormer) for audio encoder
