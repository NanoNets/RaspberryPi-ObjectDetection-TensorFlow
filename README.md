# Object Detection Using Tensorflow on the Raspberry Pi

Script for object detection from training new model on dataset to exporting quantized graph

## Step 1. Setup

### Using docker registry
This is the fastest way to use the repo
```
docker pull docker.nanonets.com/pi_training
```
OR

### Building locally
Copy dataset with `images` folder containing all training images and `annotations` folder containing all respective annotations inside data folder in repo which will be mounted by docker as volume

Ideally should run this script using nvidia-docker

#### Docker build script
Should run this script from repository root
```
docker build -t pi_training -f docker/Dockerfile.training.gpu .
docker image tag pi_training docker.nanonets.com/pi_training
```
------

## Step 2. Starting training
Tensorboard will be started at port 8000 and run in background
You can specify -h parameter to get help for docker script

```
sudo nvidia-docker run -p 8000:8000 -v `pwd`:data docker.nanonets.com/pi_training -m train -a ssd_mobilenet_v1_coco -e ssd_mobilenet_v1_coco_0 -p '{"batch_size":8,"learning_rate":0.003}'
```

### Usage
The docker instance on startup runs a script run.sh which takes the following parameters:
```
run.sh [-m mode] [-a architecture] [-h help] [-e experiment_id] [-c checkpoint] [-p hyperparameters]
```
	-h          display this help and exit
	-m          mode: should be either `train` or `export`
	-p          key value pairs of hyperparameters as json string
	-e          experiment id. Used as path inside data folder to run current experiment
	-c          applicable when mode is export, used to specify checkpoint to use for export

**List of Models (that can be passed to -a):**
1. ssd_mobilenet_v1_coco
2. ssd_inception_v2_coco
3. faster_rcnn_inception_v2_coco
4. faster_rcnn_resnet50_coco
5. rfcn_resnet101_coco
6. faster_rcnn_resnet101_coco
7. faster_rcnn_inception_resnet_v2_atrous_coco
8. faster_rcnn_nas

**Possible hyperparameters to override from -p command in json** 

| Name | Type |
|-----------|-----------------|
| learning_rate | float |
| batch_size | int |
| train_steps | int |
| eval_steps | int |

------

## Step 3. Exporting trained model
This command would export trained model in quantized graph that can be used for prediction. You need to specify one of the trained checkpoints from experiment directory that you want to use for prediction with -c command as follows:

```
sudo nvidia-docker run -v `pwd`:data docker.nanonets.com/pi_training -m export -a ssd_mobilenet_v1_coco -e ssd_mobilenet_v1_coco_0 -c /data/0/model.ckpt-8998
```

Once your done training the model and have exported it you can move this onto a client device like the Raspberry Pi.
For details of how to use on the Raspberry Pi click see https://github.com/NanoNets/TF-OD-Pi-Test
