# Long-Range-3D-Self-Attention-for-Prostate-Segmentation

Model and dataset from the paper _Long-Range 3D Self-Attention for Prostate Segmentation_ 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage

Our proposed model can be trained with the command:


```bash

main.py --dataset public_prostata --interp_size 256 
--crop_size 56 144 144 --crop_type center --loss BCE_jaccard --learning_rate 0.1 --job_id ####_##
```

To load weights of a given experiment you can specify its job id with the keyword 
--job_id ####_## 
Predictions and ground truth slices can be dump by using the flag --plot_flag

## Dataset
We used the [Prostate-MRI-US-Biopsy](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661) after a lot of pre-processing.
We included two YAML file inside the preprocessing folder so that researchers can replicate our dataset easily:

1. ```prostate_stl.yml``` contains all the selected patients along with their label as STL files;

2. ```prostate_npy.yml``` contains all the selected patients along with their label as npy files;

you can use the former to create the numpy files from meshes using the function in ```preprocessing/explore_dataset.py```, you can then use the latter to feed our dataloader. Remember to update the path with your numpy_yaml in ```yaml_segmentation_dataset.py```.

we also included other useful functions in ```preprocessing/explore_dataset.py``` to check the quality of MRI scans. 
