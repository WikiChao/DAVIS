# High-Quality Visually-Guided Sound Separation from Diverse Categories
> **Chao Huang, Susan Liang, Yapeng Tian, Anurag Kumar, Chenliang Xu**
>
> We propose DAVIS, a Diffusion-based Audio-VIsual Separation framework that solves the audio-visual sound source separation task through generative learning. Existing methods typically frame sound separation as a mask-based regression problem, achieving significant progress. However, they face limitations in capturing the complex data distribution required for high-quality separation of sounds from diverse categories. In contrast, DAVIS leverages a generative diffusion model and a Separation U-Net to synthesize separated sounds directly from Gaussian noise, conditioned on both the audio mixture and the visual information. With its generative objective, DAVIS is better suited to achieving the goal of high-quality sound separation across diverse sound categories. We compare DAVIS to existing state-of-the-art discriminative audio-visual separation methods on the AVE and MUSIC datasets, and results show that DAVIS outperforms other methods in separation quality, demonstrating the advantages of our framework for tackling the audio-visual source separation task.

<a href="https://wikichao.github.io/data/projects/DAVIS/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/pdf/2308.00122v2"><img src="https://img.shields.io/badge/arXiv-DAVIS-b31b1b.svg" height=20.5></a>

<p align="center">
<img src="asset/teaser.pdf" width="800px"/>
</p>

## Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/WikiChao/DAVIS.git
cd DAVIS

conda create --name DAVIS python=3.8
conda activate DAVIS

pip install -r requirements.txt
```

## Dataset

### 1. Download Datasets

- **MUSIC Dataset:**  
  Download from [MUSIC Dataset GitHub](https://github.com/roudimit/MUSIC_dataset).  

- **AVE Dataset:**  
  Download from [AVE Dataset GitHub](https://github.com/YapengTian/AVE-ECCV18).

---

### 2. Preprocess Videos

Preprocess the videos according to your needs, ensuring the index files are consistent.  

- **Frame Extraction:** Refer to `./preprocessing/extract_frames.py`.  
- **Audio Extraction:** Extract waveforms at 11,025 Hz. You can use `./preprocessing/extract_audio.py`.

---

### 3. Data Splits

We provide `.csv` index files for training and testing.  
The index files are located at:  
- `./data/MUSIC` for MUSIC  
- `./data/AVE` for AVE  

---

### 4. Directory Structure

The directory structure for the datasets is as follows:  


    ```
    data
    ├── audio
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.wav
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.wav
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...

    ```


## Training

We provide a minimal example to launch the training. To get started, try running:

```bash
cd scripts

bash run.sh # for MUSIC dataset

or 

bash run_ave.sh # for AVE dataset
```

## Evaluation

To launch the evaluation, modify the following arguements in ``run.sh`` or ``run_ave.sh`` to the following:

```bash
OPTS+="--split test "
OPTS+="--mode eval"
```

## Acknowledgements

We borrow code from the following repositories:

- [CCoL](https://github.com/YapengTian/CCOL-CVPR21)
- [diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) 
- [iQuery](https://github.com/JiabenChen/iQuery) 

## Citation
If you use this code for your research, please cite the following work: 
```
@article{huang2023davis,
  title={DAVIS: High-Quality Audio-Visual Separation with Generative Diffusion Models},
  author={Huang, Chao and Liang, Susan and Tian, Yapeng and Kumar, Anurag and Xu, Chenliang},
  journal={arXiv preprint arXiv:2308.00122},
  year={2023}
}
```
