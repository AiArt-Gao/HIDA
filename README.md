# DISC

## Introduction

- DISC is an open  source, PyTorch-based method for **facial sketch synthesis(FSS)**.


-  For more information about efficientSegmentation, please read the following paper:  "Depth-Informed and Style-Controllable Facial Sketch Synthesis via Dynamic Adaptation" .


- This project generates a quality and vivid sketch portrait from a given photo using a GAN-based model.  Besides, our method allows precise style control of the synthesised sketch. Please also cite this paper if you are using the method for your research! 


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

- **More Results:**

We offer more results here: https://drive.google.com/file/d/1vT0nqEVVByBW1QltYVX_mIYCcZ4wXsQD/view?usp=sharing

## Apply a pre-trained model

- A face photo↦sketch model pre-trained on dataset [FS2K](https://github.com/DengPingFan/FS2K)
- The [pre-trained model](https://drive.google.com/file/d/1vT0nqEVVByBW1QltYVX_mIYCcZ4wXsQD/view?usp=sharing) need to be save at `./checkpoint`
- Then you can test the model

### Train/Test

- Download the dataset [FS2K](https://github.com/DengPingFan/FS2K) here

- Train a model

  ```
  python train.py --root your_root_path_train
  ```

- Test the model: please prepare your test data's depth maps using 3DDFA methods

  ```
  python test.py --data_dir your_data_path_test --depth_dir your_depth_path_test 
  ```

- If you want to train on your own data, please first align your pictures and prepare your data's depth maps according to tutorial in preprocessing steps.

## Citation

 If you use this code for your research, please cite our paper. 

## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and 

[Genre]: https://github.com/fei-hdu/genre



.