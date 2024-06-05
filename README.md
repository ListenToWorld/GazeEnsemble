# GazeEnsemble
In this work, we propose a method called GazeEnsemble. This method integrates three models: GazeLSTM, GazePose, and GazeViT, and applies data augmentation to the test dataset. Ultimately, the mAP on the test set reached 0.80.

## Main Results

| Model |  Ego4D (mAP)  | 
| :-----: | :---: | 
|  Baseline  |   0.66 |
|  EgoFlow  |  0.74 |
|  GazePose |  0.79 |
|  VideoMAE-L | 0.73 |
|  GazeEnsemble | 0.80|

## Requiments
- pytorch
- opencv
- mediapipe
- numpy=1.23.5
- mmcv=2.1.0
- xgboost=2.0.3
- albumentations==1.4.2

## Quick Start

### Data preparation
You should prepare your data just as [*Ego4D*](https://github.com/EGO4D/social-interactions/tree/lam). In addition, it is also necessary to generate some JSON files according to [*GazePose*](https://github.com/lemon-prog123/GazePose) and place them under json_prior
Specially, your structure should be like this:

```
data/
* csv/
  
* json/
  
* split/
  
* result_LAM/
  
* json_original/

* json_prior/

* video_imgs/
  
* face_imgs/
  * uid
    * trackid
      * face_xxxxx.jpg
```


### Train
For GazeLSTM, you can use the following command for training:

```
python run.py --batch_size 128 --lr 1e-4 --val_stride 13 --train_stride 13 --model GazeLSTM --num_workers 12 --exp_path trainlog/gaze_lstm --checkpoint checkpoints/gaze360_model.pth
```
For GazePose, you can use the following command for training:

```
python run.py --Gaussiandrop --prior_head --prior_landmark --mesh --batch_size 128 --lr 1e-4 --val_stride 13 --train_stride 13 --model GazePose --num_workers 12 --exp_path trainlog/gaze_pose
```

For GazeViT, you can use the following command for training:

```
python run.py --Gaussiandrop --prior_head --prior_landmark --mesh --batch_size 128 --lr 1e-4 --val_stride 13 --train_stride 13 --model GazeViT --num_workers 12 --exp_path trainlog/gaze_vit --checkpoint checkpoints/GazePose.pth --pretrained checkpoints/vit-small-p16.pth
```

### Inference
For GazeLSTM, you can use the following command for inference:

```
python run.py --batch_size 128 --model GazeLSTM --num_workers 12 --exp_path validationlog/gaze_lstm --checkpoint checkpoints/gazelstm.pth --eval
```
For GazePose and GazeViT, you need to modify the values of model, exp_path, and checkpoint.

If you want to inference on the flipped test dataset, you can add '--flip' to the command.

### Ensemble Model
For training ensemble model using the training set, it is necessary to first place the CSV files of the predicted values of the participating models on the training dataset in the same folder, and place the CSV files of the predicted values on the test dataset in another folder, then run [ensemble.py](ensemble.py) to generate new predict csv file on test dataset.

### Weights of Model

| Model |  url  | 
| :-----: | :---: | 
|  vit-s  |   [*url*](https://yunpan.ustb.edu.cn/link/AAC4209827D8ED4B26BAD1471446E3C829) |
|  hopenet  |  [*url*](https://yunpan.ustb.edu.cn/link/AA9A0BD5161C6E4F96BCE05B4EA35181BD)|
|  gaze360 |  [*url*](https://yunpan.ustb.edu.cn/link/AA9F2176E9562F4FE18B30F9CC87FDCAB4) |
|  GazePose | [*url*](https://yunpan.ustb.edu.cn/link/AAD858E87924874D6D96CE45F16E786AE3) |
