# pi_docker

Script for object detection from training new model on dataset to exporting quantized graph

## Setup
Copy dataset with `images` folder containing all training images and `annotations` folder containing all respective annotations inside data folder in repo which will be mounted by docker as volume

Ideally should run this script using nvidia-docker

## Docker build script
Should run this script from repository root
```
docker build -t pi-od -f docker/Dockerfile.training.gpu .
```

## Starting training
Tensorboard will be started at port 8000 and run in background

```
sudo nvidia-docker run -p 8000:8000 -v `pwd`:data:data nanonets.docker.com/pi-od -m train -a ssd_mobilenet_v1_coco -e ssd_mobilenet_v1_coco -p '{"batch_size":8,"learning_rate":0.003}'
```