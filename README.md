# HIDA: Human-Inspired Facial Sketch Synthesis with Dynamic Adaptation

## Abstract
Facial sketch synthesis (FSS) aims to generate a vivid sketch portrait from a given facial photo.    Existing FSS methods merely rely on 2D representations of facial semantic or appearance. However, professional human artists usually use outlines or shadings to covey 3D geometry. Thus facial 3D geometry (e.g. depth map) is extremely important for FSS. Besides, different artists may use diverse drawing techniques and create multiple styles of sketches; but the style is globally consistent in a sketch. Inspired by such observations, in this paper, we propose a novel \textit{Human-Inspired Dynamic Adaptation} (HIDA) method. Specially, we propose to dynamically modulate neuron activations based on a joint consideration of both facial 3D geometry and 2D appearance, as well as globally consistent style control. Besides, we use deformable convolutions at coarse-scales to align deep features, for generating abstract and distinct outlines. Experiments show that HIDA can generate high-quality sketches in multiple styles, and significantly outperforms previous methods, over a large range of challenging faces. Besides, HIDA allows precise style control of the synthesized sketch, and generalizes well to natural scenes. Our code will be released after peer review. 

## Paper Information

Fei Gao, Yifan Zhu, Chang Jiang, Nannan Wang, Human-Inspired Facial Sketch Synthesis with Dynamic Adaptation, Proceedings of the International Conference on Computer Vision (ICCV), 7237--7247, 2023.

## Citation

 If you use this code for your research, please cite our paper. 

```
@inproceedings{gao2023human,
  title={Human-Inspired Facial Sketch Synthesis with Dynamic Adaptation},
  author={Gao, Fei and Zhu, Yifan and Jiang, Chang and Wang, Nannan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7237--7247},
  year={2023}
}
```

## Pipeline

![localFace1](https://github.com/AiArt-HDU/DISC/blob/main/images/fig_pipeline.jpg)

![localFace1](https://github.com/AiArt-HDU/DISC/blob/main/images/fig_ida.jpg)

## Sample Results

- **Comparison with SOTAs on the FS2K dataset:**

![localFace1](https://github.com/AiArt-HDU/DISC/blob/main/images/localFace1.jpg)

![localFace2](https://github.com/AiArt-HDU/DISC/blob/main/images/localFace2.jpg)

![localFace3](https://github.com/AiArt-HDU/DISC/blob/main/images/localFace3.jpg)

(a)Photo   (b)Depth   (c)Ours   (d)Pix2PixHD   (e)FSGAN   (f)SCA-GAN   (g)GT   (h)Pix2Pix   (i)MDAL   (j)CycleGAN   (k)GENRE

- **Performance on faces in-the-wild:**

![wildFace1](https://github.com/AiArt-HDU/DISC/blob/main/images/wildFace1.jpg)

![wildFace2](https://github.com/AiArt-HDU/DISC/blob/main/images/wildFace2.jpg)

![wildFace3](https://github.com/AiArt-HDU/DISC/blob/main/images/wildFace3.jpg)

(a)Photo   (b)Ours(Style1)   (c)Ours(Style2)   (d)Ours(Style3)   (e)GENRE   (f)Pix2Pix   (g)CycleGAN   (h)SCA-GAN

-  **Performance of our DISC model on natural images:**

![cat](https://github.com/AiArt-HDU/DISC/blob/main/images/cat.png)

![building1](https://github.com/AiArt-HDU/DISC/blob/main/images/building1.png)

![building2](https://github.com/AiArt-HDU/DISC/blob/main/images/building2.png)

(a)Photo   (b)Depth   (c)Ours(Style1)   (d)Ours(Style2)   (e)Ours(Style3)

- **Extension to Pen-drwings and Oilpaintings**
![localFace1](https://github.com/AiArt-HDU/DISC/blob/main/images/fig_penoil.jpg)

- **More Results:**

We offer more results here: https://drive.google.com/file/d/1vT0nqEVVByBW1QltYVX_mIYCcZ4wXsQD/view?usp=sharing

## Prerequisites

- Linux or macOS
- Python 3.8.12
- Pytorch-lightning 0.7.5
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

  ```
  git clone https://github.com/AiArt-HDU/DISC
  cd DISC
  ```

- Install PyTorch 1.7.1 and torchvision from [http://pytorch.org](http://pytorch.org/) and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by

  ```
  pip install -r requirements.txt
  ```

- The installation environment of DCN-V2 depandency is more complicated，you can refer to the 

  [official installation method]: https://github.com/EMI-Group/FaPN

## Apply a pre-trained model

- A face photo↦sketch model pre-trained on dataset [FS2K](https://drive.google.com/file/d/1ATeDl-Vu2Ztq3i0jeu9Qyap0MyOdg05v/view?usp=sharing)
- The [pre-trained model](https://drive.google.com/file/d/1q9S7nHaweH8OMmfVZ9Zv4FcHNHjfBsDy/view?usp=sharing) need to be save at `./checkpoint`
- Then you can test the model

### Train/Test

- Download the dataset [FS2K](https://drive.google.com/file/d/1ATeDl-Vu2Ztq3i0jeu9Qyap0MyOdg05v/view?usp=sharing) here

- Train a model

  ```
  python train.py --root your_root_path_train
  ```

- Test the model: please prepare your test data's depth maps using 3DDFA methods

  ```
  python test.py --data_dir your_data_path_test --depth_dir your_depth_path_test 
  ```

- If you want to train on your own data, please first align your pictures and prepare your data's depth maps according to tutorial in **preprocessing steps**.

### Preprocessing steps

Face photos (and paired drawings) need to be aligned and have depth maps. And depth maps after alignment are needed in our code in training.

In our work, depth map is generated by method in [1]

- First, we need to align, resize and crop face photos (and corresponding drawings) to 250x250
- Then,we use code in [3DDFA](https://github.com/cleardusk/3DDFA_V2)

 to generate depth maps for face photos and drawings.

*[1] J. Guo, X. Zhu, Y. Yang, F. Yang, Z. Lei, and S. Z. Li, “Towards fast, accurate and stable 3d dense face alignment,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020.*

## Citation

 If you use this code for your research, please cite our paper. 

## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [GENRE](https://github.com/fei-hdu/genre), and [CocosNet](https://github.com/microsoft/CoCosNet).

