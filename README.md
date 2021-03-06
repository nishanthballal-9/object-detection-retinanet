# Object Detection RetinaNet (Person, Car)

![image_000000086](https://github.com/nishanthballal-9/object-detection-retinanet/blob/main/images/image_000000086.jpg)
![image_000000183](https://github.com/nishanthballal-9/object-detection-retinanet/blob/main/images/image_000000183.jpg)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.

## Results
Currently, this repo achieves 28.0% mAP at 600px resolution with a Resnet-101 backbone.

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests

```

## Train Test Split

Train test split is achieved using cocosplit.py. The files train.json and val.json can be found in trainval/annotations.

Replace images in trainval by the images folder obtained from:
-https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz

Create train and test folders by running trainval/data_split.py

## Training

The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

```
python train.py --dataset coco --coco_path trainval --depth 50
```

Performed training for depth 50 (100 epochs) and depth 101 (50 epochs).

Link for trained model:
1. ResNet50 backbone - https://drive.google.com/file/d/1Th5NakykxoiSnEDhNYYsUmH1oU0wBeyl/view?usp=sharing
2. ResNet101 backbone - https://drive.google.com/file/d/1xHFV5QITGLsKfxHhaHNzxnozwhLfOzgp/view?usp=sharing

## Validation

Run `coco_validation.py` to validate the code on the COCO dataset. With the above model, run:

`python coco_validation.py --coco_path ~/path/to/coco --model_path /path/to/model/model_depth101.pt`

This produces the following results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.229

```

## Visualization

To visualize the network detection, use `visualize_single_image.py`:

```
python visualize_single_image.py --image_dir <path/to/val/images> --model <path/to/model_depth101.pt> --class_list class_mapping.csv
```

## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
person,0
car,1
```

## Acknowledgements

- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)

## Examples

![image_000000208](https://github.com/nishanthballal-9/object-detection-retinanet/blob/main/images/image_000000208.jpg)
![image_000000278](https://github.com/nishanthballal-9/object-detection-retinanet/blob/main/images/image_000000278.jpg)
![image_000001293](https://github.com/nishanthballal-9/object-detection-retinanet/blob/main/images/image_000001293.jpg)

