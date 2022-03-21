# Track, Check, Repeat: An EM Approach to Unsupervised Tracking

This is the official code release for our CVPR21 paper on unsupervised detection and tracking. It produces results slightly better than reported in the paper.


**[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Harley_Track_Check_Repeat_An_EM_Approach_to_Unsupervised_Tracking_CVPR_2021_paper.pdf)] [[Project Page](http://www.cs.cmu.edu/~aharley/em_cvpr21/)]**


<img src='http://cs.cmu.edu/~aharley/images/em_loop_kitti.gif'>

**We use ensemble agreement between 2d and 3d models, as well as motion cues, to unsupervisedly learn object detectors from scratch.** Top: 3d detections. Middle: 2d segmentation. Bottom-left: unprojected 2d segmentation, in a bird's eye view. Bottom-right: 3d detections, in a bird's eye view. 


### Overview

<img src="http://www.cs.cmu.edu/~aharley/em_cvpr21/images/fig1.png" width="800px"/>

**An EM approach to unsupervised tracking.** We present an expectation-maximization (EM) method, which takes RGBD
videos as input, and produces object detectors and trackers as output. (a) We begin with a handcrafted E step, which uses optical flow
and egomotion to segment a small number of objects moving independently from the background. (b) Next, as an M step, we treat these
segmented objects as pseudo-labels, and train 2D and 3D convolutional nets to detect these objects under heavy data augmentation. (c) We
then use the learned detectors as an ensemble to re-label the data (E step), and loop.


### Preparation

```
pip install -r requirements.txt
```
Also, download a [RAFT](https://github.com/princeton-vl/RAFT) model, and put its path into `tcr_kitti_discover.py` and `tcr_kitti_ensemble.py`. In this repo, we use [raft-sintel.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT). Code for running RAFT is already included in this repo (in `nets/raftnet.py` and `nets/raft_core`).

### Data

In this repo we focus on the KITTI dataset. We use sequences 0000-0008 for training, and sequences 0009-0010 for testing. If you are familiar with KITTI, you can swap out `simplekittidataset.py` with your own dataloader.

To use our dataloader, download [kitti_prep.tar.gz](https://drive.google.com/file/d/1g1tCuTB4jSON4NLBP28Qi1FNDLHGhcby/view?usp=sharing) (12g), untar it somewhere, and update the `root_dir` in `simplekittidataset.py`. 

Our prepared npzs include (up to) 1000 2-frame snippets from each sequence. (Some sequences are less than 1000 frames long.) Each npz has the following data:
```
rgb_camXs # S,3,H,W = 2,3,256,832; uint8; image data from color camera 2
xyz_veloXs # S,V,3 = 2,130000,3; float32; velodyne pointcloud
cam_T_velos # S,4,4 = 2,4,4; float32; 4x4 that moves points from velodyne to camera
rect_T_cams # S,4,4 = 2,4,4; float32; 4x4 that moves points from camera to rectified 
pix_T_rects # S,4,4 = 2,4,4; float32; 4x4 that moves points from rectified to pixels
boxlists # S,N,9 = 2,50,9; float32; 3d box info: cx,cy,cz,lx,ly,lz,rx,ry,rz
tidlists # S,N = 2,50; int32; unique ids
clslists # S,N = 2,50; int32; classes
scorelists # S,N = 2,50; float32; valid mask for boxes
```
We use `S` for sequence length, `H,W` are for image height and width, `V` is the maximum number of points in the velodyne pointcloud, and `N` is the maximum number of annotated objects.

In this data, and throughout the code, we use the following convention:

- `p_a` is a point named `p` living in `a` coordinates.
- `a_T_b` is a transformation that takes points from coordinate system `b` to coordinate system `a`.

For example, `p_a = a_T_b * p_b`.

### Pre-trained model

Download a pre-trained model here: [model-000050000.pth](https://drive.google.com/file/d/100sG4w2T0OO9mQgchyTWlUECNRxtgM6b/view?usp=sharing) (19.4mb).

This model is the result of 5 rounds of EM, trained in this repo. It outperforms the best model reported in the paper. 

#### BEV 

mAP@IOU | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 
--- | --- | --- | --- |--- |--- |--- |--- 
Paper | 0.40 | 0.38 | 0.35 | 0.33 | 0.31 | 0.23 | 0.06
This repo | **0.58** | **0.58** | **0.54** | **0.48** | **0.42** | **0.29** | **0.10**

#### Perspective 

mAP@IOU | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 
--- | --- | --- | --- |--- |--- |--- |--- 
Paper | 0.43 | 0.4 | 0.37 | 0.35 | 0.33 | 0.30 | **0.21**
This repo | **0.60** | **0.59** | **0.59** | **0.56** | **0.49** | **0.23** | 0.10


### Differences from the paper

The method in this repo is slightly different and easier to "get working" than the method in the paper.

First, we improve the model in two ways: 

- **We use 2d BEV convs in the 3d CenterNet model, instead of 3d convs.** This allows higher resolution and a deeper model.

- **We estimate 3d segmentation in addition to center-ness, and use this in the ensemble.** The connected components step depends on accurate shape estimation, so the 3d segmentation here makes more sense than the center-ness heatmap, and leads to finding boxes that are much closer to ground truth. We implement this simply by adding new channels to the CenterNet.

- **We replace the additive ensemble with a multiplicative one.** Previously, we added together all our signals (independent motion, 2d objectness, 3d objectness, occupancy), and tuned a threshold on this sum. Now, we multiply everything together: a region is only considered an object if (1) it is moving independently, *and* (2) the 2d segmentor fired, *and* (3) the 3d segmentor fired, *and* (4) it is occupied. This gives lower recall but higher precision, and it reduces drift.

Also, we simplify the model, by eliminating a hyperparameter:

- **We eliminate center-surround saliency scoring/thresholding** (Equation 7 in the CVPR paper). Previously, center-surround saliency was critical for discarding pseudo-labels that cut an object in half. In the current version (thanks to the multiplicative ensemble and segmentation estimates), this happens less frequently, and so we are able to remove it. This is good news, because tuning the center-surround saliency threshold was tricky in the original work. Note that a type of center-surround still exists, since we only take objects that are moving independently from their background.


### Training from scratch

Initial E step: 

```
py tcr_kitti_discover.py --output_name="stage1"
```
This will output "stage1" npzs into `tcr_pseudo/`.

M step: 
```
py tcr_kitti_train_2d.py --input_name="stage1" --max_iters=10000
py tcr_kitti_train_3d.py --input_name="stage1" --max_iters=10000
```
This will train 2d and 3d models on those pseudo-labels, and save checkpoints into `checkpoints/`. The 3d model will also occasionally evaluate itself against ground truth. 

E step: 
```
py tcr_kitti_ensemble.py \
   --init_dir_2d=your_2d_model_folder \
   --init_dir_3d=your_3d_model_folder \
   --log_freq=1 \
   --shuffle=False \
   --export_npzs=True \
   --output_name="stage2" \
   --exp_name="go" \
   --skip_to=0 \
   --max_iters=3837
```
This will use the 2d and 3d models along with motion cues to output "stage2" npzs into `tcr_pseudo/`:

Then repeat the M and E steps, updating the input/output names as the stages progress. We find it is helpful in later stages to train for longer (e.g., 20k or 50k iterations, instead of 10k.)

### Evaluation

```
py tcr_kitti_eval.py --init_dir_3d=your_3d_model_folder
```



### Citation

If you use this code for your research, please cite:

**Track, Check, Repeat: An EM Approach to Unsupervised Tracking**.
[Adam W. Harley](https://cs.cmu.edu/~aharley),
[Yiming Zuo](https://zuoym15.github.io/),
[Jing Wen](https://wenj.github.io/),
[Shubhankar Potdar](http://smpotdar.com),
Ritwick Chaudhry,
[Katerina Fragkiadaki](http://cs.cmu.edu/~katef/). In CVPR 2021.

Bibtex:
```
@inproceedings{harley2021track,
  title={Track, check, repeat: An EM approach to unsupervised tracking},
  author={Harley, Adam W and Zuo, Yiming and Wen, Jing and Mangal, Ayush and Potdar, Shubhankar and Chaudhry, Ritwick and Fragkiadaki, Katerina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

