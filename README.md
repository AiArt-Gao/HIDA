# DISC

## Introduction

- DISC is an open  source, PyTorch-based method for **facial sketch synthesis(FSS)**.


-  For more information about efficientSegmentation, please read the following paper:  "Depth-Informed and Style-Controllable Facial Sketch Synthesis via Dynamic Adaptation" .


- This project generates a quality and vivid sketch portrait from a given photo using a GAN-based model.  Besides, our method allows precise style control of the synthesised sketch. Please also cite this paper if you are using the method for your research! 


## Sample Results

- **Comparison with SOTAs on the FS2K dataset:**

![1663311368105](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663311368105.png)

![1663311033413](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663311033413.png)

![1663299824768](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663299824768.png)

**(a)**Photo   **(b)**Depth   **(c)**Ours   **(d)**Pix2PixHD   **(e)**FSGAN   **(f)**SCA-GAN   **(g)**GT   **(h)**Pix2Pix   **(i)**MDAL   **(j)**CycleGAN   **(k)**GENRE

- **Performance on faces in-the-wild:**

![1663311795153](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663311795153.png)

![1663311928092](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663311928092.png)

![1663311845852](C:\Users\29616\AppData\Roaming\Typora\typora-user-images\1663311845852.png)

**(a)**Photo   **(b)**Ours(Style1)   **(c)**Ours(Style2)   **(d)**Ours(Style3)   **(e)**GENRE   **(f)**Pix2Pix   **(g)**CycleGAN   **(h)**SCA-GAN

-  **Performance of our DISC model on natural images:**

![](I:\D盘\zyf\2020研究生\毕业论文\2021阶段性工作总结\结果图汇总\scene\scene3\5_1.png)

![](I:\D盘\zyf\2020研究生\毕业论文\2021阶段性工作总结\结果图汇总\scene\buildings\4_2.png)

![](I:\D盘\zyf\2020研究生\毕业论文\2021阶段性工作总结\结果图汇总\scene\buildings\3_2.png)

**(a)**Photo   **(b)**Depth   **(c)**Ours(Style1)   **(d)**Ours(Style2)   **(e)**Ours(Style3)

- **More Results:**

We offer more results here: https://pan.baidu.com/s/1SoaarTYejTMNT_V4agGDWg ( Extraction code : j2e3 )

## Citation

 If you use this code for your research, please cite our paper. 

## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and 

[Genre]: https://github.com/fei-hdu/genre



.