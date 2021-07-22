# tensorflow2-yolov5
YoloV5 implemented by TensorFlow2 , with support for training, evaluation and inference.


## Table of Contents
* [Data Preparation](#data-preparation)
<!-- * [License](#license) -->

## Data Preparation

### Download VOC
```
$ bash preparation/get_voc.sh
```
### Generate txt file 
```
$ cd preprocess
$ python prepare_data.py --class_name_dir '../preparation/voc.names' --output_dir '../preparation/txt_files/voc'
```

