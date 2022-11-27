# CSPNet
This repository contains the code for the paper "Deep Implicit Surface Point Prediction Networks" [[PDF]](https://arxiv.org/abs/2106.05779)[[Project page]](https://sites.google.com/view/cspnet).

You can find detailed usage instructions for training your own models and using pretrained models below.

If you find our code or paper useful, please consider citing
```
    @InProceedings{Venkatesh_2021_ICCV,
    author    = {Venkatesh, Rahul and Karmali, Tejan and Sharma, Sarthak and Ghosh, Aurobrata and Babu, R. Venkatesh and Jeni, L\'aszl\'o A. and Singh, Maneesh},
    title     = {Deep Implicit Surface Point Prediction Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12653-12662}
}
```

## Installation
### Environment
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `mesh_funcspace` using
```
conda env create -f environment.yaml
conda activate mesh_funcspace
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

To compile the dmc extension, you have to have a cuda enabled device set up.
If you experience any errors, you can simply comment out the `dmc_*` dependencies in `setup.py`.
You should then also comment out the `dmc` imports in `im2mesh/config.py`.

### Dataset
To evaluate a pretrained model or train a new model from scratch, you have to obtain the dataset.
You can download the ShapeNet dataset and run the preprocessing pipeline as explained next.

Take in mind that running the preprocessing pipeline yourself requires a substantial amount time and space on your hard drive.


#### Building the dataset
Alternatively, you can also preprocess the dataset yourself.
To this end, you have to follow the following steps:
* download the [ShapeNet dataset v1](https://www.shapenet.org/) and put into `data/external/ShapeNet`. 
* download the [renderings and voxelizations](http://3d-r2n2.stanford.edu/) from Choy et al. 2016 and unpack them in `data/external/Choy2016` 
* build our modified version of [mesh-fusion](https://github.com/davidstutz/mesh-fusion) by following the instructions in the `external/mesh-fusion` folder

You are now ready to build the dataset:
```
cd scripts
bash dataset_shapenet/build.sh
``` 

This command will build the dataset in `data/ShapeNet_full.build`.
To install the dataset, run
```
bash dataset_shapenet/install.sh
```

If everything worked out, this will copy the dataset into `data/ShapeNet_full`.

### Evaluation
You can run evaluation using
```
python eval.py --config CONFIG.yaml
```
The script will generate Chamfer distance, Normal Cosine Distance
(fwd. and jac.), Sillhouette IoU and Depth Error.

Pretrained Model: [link](https://huggingface.co/tejank10/cspnet/resolve/main/model_best.pt)

### Training
To train a new network from scratch, run
```
python train.py --config CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.

## Acknowledgement
This codebase is built upon the [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).
