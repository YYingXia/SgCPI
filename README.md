# SgCPI
Here, we propose a subgraph-based model SgCPI to incorporate local known interacting network to predict CPI interactions for unseen proteins and compounds. SgCPI first randomly samples the local CPI network of the query compound-protein pair as a subgraph with better computational efficiency, and then applies a heterogeneous graph neural network (HGNN) to embed the active and inactive message of the subgraph with the physicochemical properties. To further improve the generalizability for unseen compounds and proteins, we design SgCPI-KD, which uses SgCPI as a teacher model to distillate the knowledge of SgCPI by estimating the potential neighbors and integrating them with molecular features.
## Preparation
SgCPI is built on Python3.
We recommend to use a virtual environment for the installation of SgCPI and its dependencies.
A virtual environment can be created and (de)activated as follows by using conda:

    # create
    $ conda create -n SgCPI_env python=3.8
    # activate
    $ source activate SgCPI_env

When you want to quit the virtual environment, just:

    $ source deactivate

Download the source code from GitHub:

    $ git clone https://github.com/yingx/SgCPI.git

Download the dataset and molecular features from [url](http://www.csbio.sjtu.edu.cn/bioinf/SgCPI/files/dataset.tar.gz).

    $ tar zxvf dataset.tar.gz
    $ cp -r dataset SgCPI

Install the dependencies as following:

    torch==1.8.1
    torch-scatter==2.0.8
    torch-sparse==0.6.12
    torch-geometric==2.0.3
    rdkit==2022.03.2
    biopython==1.81
    scikit-learn==1.3.0
    joblib==1.3.1


## Usage

### Method 1. SgCPI
SgCPI can be applied for transductive, semi-inductive and inductive settings.
Running the results for the fold 0 of inductive setting:

    $ python main_SgCPI.py --setting semi-inductive --fold 0


### Method 2. SgCPI-KD
SgCPI can be applied for the inductive settings.
Running the results for the fold 0 of inductive setting:

    $ python main_SgCPI_KD.py --fold 0


## License
All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: [CCBY4.0](https://github.com/YYingXia/SgCPI/blob/main/LICENSE).

## Online service
Online retrieval service and benchmark datasets are in [here](http://www.csbio.sjtu.edu.cn/bioinf/SgCPI/).
