# ACM Sigmod Programming Contest - Team WBSG

This repository contains code and relevant instructions to reproduce our final submission for the [ACM Sigmod Programming Contest 2022](http://sigmod2022contest.eastus.cloudapp.azure.com/index.shtml).

## Blocking System

* **Requirements**

    [Pip](https://pypi.org/project/pip/) or [Anaconda3](https://www.anaconda.com/products/individual)


* **Building the python environment using pip**

    To build the python environment used for the submissions, navigate to the project root folder where the file *requirements.txt* is located and run `pip3 install -r requirements.txt`.


* **Reproducing our final submission**

    To reproduce our final submission, please run `python3 blocking_neural.py` in the root folder.
    The script produces the requested `output.csv`, which is used for the evaluation of the blocking system.

    
## (Optional) Training the contrastive model

The checkpoints of the trained models are included in this repository in the `models/` folder. It is not necessary to rerun the training for the blocking system to work. If you specifically want to replicate the training, follow these steps:

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

* **Building the conda environment**

    To build the conda environment used for the challenge, navigate to the project root folder where the file *sigmod.yml* is located and run `conda env create -f sigmod.yml`

* **Install project as package**

	You need to install the project as a package. To do this, activate the environment with `conda activate sigmod`, navigate to the root folder of the project, and run `pip install -e .`

* **Downloading the raw data files**

    Navigate to the `src/data/` folder and run `python download_datasets.py` to automatically download the needed additional files into the correct locations.
    You can find the data at `data/raw/`

* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. `src/processing/preprocess_corpus.py`
    2. `src/processing/preprocess_sigmod.py`

* **Running the contrastive pre-training**

    Navigate to `src/contrastive/`
	
	Run the following scripts to replicate the models trained for the challenge datasets:
	
	X1: `bash sigmod1/run_pretraining_with_additional.sh microsoft/xtremedistil-l6-h256-uncased False 1024 1e-04 0.07 200 computers_only_new_15_train_sigmod_28 GPU_ID`

	X2: `bash sigmod2/run_pretraining_with_additional.sh microsoft/xtremedistil-l6-h256-uncased False 1024 5e-04 0.07 200 computers_only_new_15_train_sigmod_24 GPU_ID`
	
	You can then find the trained models in the folder `reports/`
