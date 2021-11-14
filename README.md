# Extended Pseudo 4DCNN
## High-fidelity View Synthesis for Light Field Imaging with Extended Pseudo 4DCNN
### Authors: Yunlong Wang, Fei Liu, Kunbo Zhang, Zilei Wang, Zhenan Sun, Tieniu Tan.
@ARTICLE{9061053,
  author={Wang, Yunlong and Liu, Fei and Zhang, Kunbo and Wang, Zilei and Sun, Zhenan and Tan, Tieniu},
  journal={IEEE Transactions on Computational Imaging}, 
  title={High-fidelity View Synthesis for Light Field Imaging With Extended Pseudo 4DCNN}, 
  year={2020},
  volume={6},
  number={},
  pages={830-842},
  doi={10.1109/TCI.2020.2986092}}

[IEEE Explore Webpage](https://ieeexplore.ieee.org/document/9061053)

<!-- @InProceedings{10.1007/978-3-030-01216-8_21,
author="Wang, Yunlong
and Liu, Fei
and Wang, Zilei
and Hou, Guangqi
and Sun, Zhenan
and Tan, Tieniu",
title="End-to-End View Synthesis for Light Field Imaging with Pseudo 4DCNN",
booktitle="Computer Vision -- ECCV 2018",
year="2018",
publisher="Springer International Publishing",
pages="340--355",
isbn="978-3-030-01216-8"
}

[webpage](https://link.springer.com/chapter/10.1007/978-3-030-01216-8_21#citeas) -->

# Installation

## Dependencies
- [x] [Python](https://www.python.org)
- [x] [keras](https://keras.io/)
- [x] [scikit-image](http://scikit-image.org/)
- [x] [scipy](https://www.scipy.org/)
- [x] [numpy](http://www.numpy.org/)
- [x] [h5py](http://www.h5py.org/)
- [x] [argparse](https://docs.python.org/3/library/argparse.html)

These packages can be also installed as `pip install -r requirements.txt`

## Generate Training and Validation datasets
* `python generate_ExtendedP4DCNN_Train_Val_Datasets.py`

## Train from Scratch
* `train_6L_SPS_ESP.py`

## Test LF images in a folder
* `test.py -D your_data_dir -M dir_to_model -S dir_to_save_results --ext img_extensions --factor UPSAMPLING_FACTOR --crop_length LENGTH --angular_size IMG_ANGULAR_RESOLUTION --save_results YES`

## Pre-trained Models and Utilities
* Some pre-trained models are under `model` folder, some metric calculating toolkits are under `utils` folder.
