# PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network

### Introduction
This is a tensorflow re-implementation of [PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1806.02559).

Thanks for the author's ([@whai362](https://github.com/whai362)) awesome work!

### Installation
1. Any version of tensorflow version > 1.0 should be ok.

### Download
1. Models trained on ICDAR 2017 (training set) + ICPR 2018 (training set): be avariable
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=8 --checkpoint_path=./resnet_v1_50_rbox/ \
--training_data_path=./data/ocr/icdar2015/
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note:**
1. right now , only support icdar2017 data format input, like (116,1179,206,1179,206,1207,116,1207,"###"),
but you can modify data_provider.py to support polygon format input
2. Already support polygon shrink by using pyclipper module
3. this re-implemention is just for fun, but I'll continue to improve this code.

### Test
run
```
python eval.py --test_data_path=./tmp/images/ --gpu_list=0 --checkpoint_path=./resnet_v1_50_rbox/ \
--output_dir=./tmp/
```

a text file and result image will be then written to the output path.


### Examples
be avariable


Please let me know if you encounter any issues(OCR group qq: 785515057).
