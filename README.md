# Source-free Depth for Object Pop-out, ICCV'23

Official PyTorch implementaton of ICCV'23 paper [Source-free Depth for Object Pop-out](https://arxiv.org/pdf/2212.05370.pdf)

<img src="https://github.com/Zongwei97/PopNet/blob/main/Imgs/popout.png"  width="500" />


# Abstract

Depth cues are known to be useful for visual perception. However, direct measurement of depth is often impracticable. Fortunately, though, modern learning-based methods offer promising depth maps by inference in the wild. In this work, we adapt such depth inference models for object segmentation using the objects' pop-out prior in 3D. The pop-out is a simple composition prior that assumes objects reside on the background surface. Such compositional prior allows us to reason about objects in the 3D space. More specifically, we adapt the inferred depth maps such that objects can be localized using only 3D information. Such separation, however, requires knowledge about contact surface which we learn using the weak supervision of the segmentation mask. Our intermediate representation of contact surface, and thereby reasoning about objects purely in 3D, allows us to better transfer the depth knowledge into semantics.  The proposed adaptation method uses only the depth model without needing the source data used for training, making the learning process efficient and practical. Our experiments on eight datasets of two challenging tasks, namely camouflaged object detection and salient object detection, consistently demonstrate the benefit of our method in terms of both performance and generalizability. 

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/network.png)


# Training/Testing Datasets

The RGB-D datasets with GT depth can be found at [SPNet](https://github.com/taozh2017/SPNet).

The COD dataset with source-free depth can downloaded from here ([Training](https://drive.google.com/file/d/1z903IE3fQderj_ngOi1rIsnTDhT7NHDS/view?usp=sharing)/[Testing](https://drive.google.com/file/d/1xd_Pe4oQZJqHX5NHSswwGk7AoaeH38eQ/view?usp=sharing))


# Train and Test

Please follow the training, inference, and evaluation steps:

```
python train.py
python test_produce_maps.py
python test_evaluation_maps.py
```
Make sure that you have changed the path to your dataset in the [config file](https://github.com/Zongwei97/PopNet/blob/main/Code/utils/options.py) and in the abovementioned Python files.

We use the same evaluation protocol as [here](https://github.com/taozh2017/SPNet/blob/main/test_evaluation_maps.py)


# Results 

### RGB-D SOD

Our results for RGB-D salient object detection (SOD) benchmarks can be downloaded here ([Google Drive](https://drive.google.com/file/d/1lyVTH_MhLxYam6Xr0WKsoa3SsMLwIs4W/view?usp=sharing)).

##### Quantitative comparison

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/SOD.png)

##### Qualitative comparison

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/rgbd.png)


### COD

Our results for camouflaged object detection (COD) benchmarks can be downloaded here ([Google Drive](https://drive.google.com/file/d/1m8Ht5A4uzvmvSXhn8hEfMJeam7pvaoia/view?usp=sharing)).

The checkpoint can be downloaded here ([Google Drive](https://drive.google.com/file/d/103FbjqVvmpoArA1ubd3f8vxiZYuUzbjH/view?usp=sharing)).

##### Quantitative comparison

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/SOD.png)

##### Qualitative comparison

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/results.png)


## Towards urban applications

We take the pretrained/freezed COD ckpt and figure out that our method can also generalize well on nightlight urban scenes:

![abstract](https://github.com/Zongwei97/PopNet/blob/main/Imgs/freezed.png)

# Citation

If you find this repo useful, please consider citing:

```
@INPROCEEDINGS{wu2023popnet,
  title={Source-free depth for object pop-out},
  author={Wu, Zongwei and Paudel, Danda Pani and Fan, Deng-Ping and Wang, Jingjing and Wang, Shuo and Demonceaux, Cédric and Timofte, Radu and Van Gool, Luc},
  booktitle={ICCV}, 
  year={2023},
}
  
```

# Related works
- ACMMM 23 - Object Segmentation by Mining Cross-Modal Semantics [[Code](https://github.com/Zongwei97/XMSNet)]
- TIP 23 - HiDANet: RGB-D Salient Object Detection via Hierarchical Depth Awareness [[Code](https://github.com/Zongwei97/HIDANet)]
- 3DV 22 - Robust RGB-D Fusion for Saliency Detection [[Code](https://github.com/Zongwei97/RFnet)]
- 3DV 21 - Modality-Guided Subnetwork for Salient Object Detection [[Code](https://github.com/Zongwei97/MGSnet)]
