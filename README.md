# GreenSwarm: An Energy-Efficient Framework for Swarm Learning with Parameter Circulation
GreenSwarm is a framework for energy-efficient collaborative machine learning that employs parameter circulation among nodes rather than parameter aggregation as is performed in conventional federated learning and swarm learning frameworks.

This repository provides simulations for centralized, typical swarm learnig, and GreenSwarm networks. 
For these simulations, neural networks have been employed to train on eight diverse classification datasets.

## Options:

### Dataset:
| Dataset  | Option |
| ------------- | ------------- |
| CIFAR-10  | --dataset cifar10  |
| Fashion_MNSIT  | --dataset fashion  |
|  Malaria | --dataset malaria  |
|  Retinal_OCT | --dataset retinal  |
| Intel Natural Scenes | --dataset intel  |
| Chest X-ray | --dataset xray  |
| IMDB Reviews  | --dataset imdb  |
| Reuters Newswire  | --dataset reuters  |

### Mode:
| Mode  | Option |
| ------------- | ------------- |
| Centralized  | --mode cent  |
| Typical Swarm Learning  | --mode fedavg  |
|  GreenSwarm | --mode gs  |

## Dataset Preparation:
The CIFAR10, Fashion_MNIST, IMDB reviews, and Reuters newswire datasets will be automatically downloaded by the program. For the other four datasets, the user should download the datasets prior to running the code and put it in a folder named **datasets** in the same directory as the app files.
Here are the links to download each dataset and the directories that should be created for each:

| Dataset  | Download Link | Path |
| ------------- | ------------- | ------------- |
| Malaria  | https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria  | ./datasets/malaria |
| Retinal_OCT  | https://www.kaggle.com/datasets/paultimothymooney/kermany2018  | ./datasets/retinal/OCT2017/train <br> ./datasets/retinal/OCT2017/test ./datasets/retinal/OCT2017/val |
|  Intel Natural Scenes | https://www.kaggle.com/datasets/puneet6060/intel-image-classification  | ./datasets/intel/seg_train <br> ./datasets/intel/seg_test |
|  Chest X-ray | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  | ./datasets/xray/train <br> ./datasets/xray/val <br> ./datasets/xray/test |


**Note**: Each folder (for example 'train') contains folders inside for each class. The classes for each of the datasets are as follows:
| Dataset  | Classes (folders) |
| ------------- | ------------- |
| Malaria | Parasitized, Uninfected | 
| Retinal_OCT | NORMAL, DRUSEN, DME, CNV| 
| Intel Natural Scenes | street, sea, mountain, glacier, forest, buildings| 
| Chest X-ray | NORMAL, PNEUMONIA | 

## Run:
To run the simulation, simply execute the run.py file with the **dataset** and **mode** arguments.
For example, to run the simulation for the CIFAR10 dataset using the GreenSwarm framework, type in:

```
python3 run.py --dataset cifar10 --mode gs

```
