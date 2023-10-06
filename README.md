• Motivation
My motivation for choosing this project is because I would like to gain experience creating multimodals with Federated Learning using IoT data.
I'm interesting in analyzing plant health in grow operations using image and various environemntal sensor data. Perhaps this project will lend me some applicable knowledge to such.

• Design goals
Create multimodal FedAvg algorithm to aggregate local autoencoders trained on various data modalities.

• Deliverables
Demonstrate that training multiple modalities with federated learning on IoT data improves classification performance by benchmarking different multimodal datasets.

• System blocks
Dataset -> Semi-supervised Federated Learning Training -> Multimodal

• Hardware and software requirements
This is a software focused project with various required libraries. A functional CPU or ideally GPU is the only hardware requirement.

• Team members responsibilities
Zachary Kashef - All the responsibility

• References
[Multimodal Federated Learning on IoT Data](https://arxiv.org/abs/2109.04833).

# Directory structure
* *src*: source code
* *data*: pre-processed data
* *config*: configuration files for experiments
* *sub*: pbs files for job submissions to clusters
* *results*: experimental results
* *plots*: data visualizations of results

# Prerequisites
To fully re-produce the results and plots in the paper, please use the provided
configuration files to run replicates of individual experiments in parallel on a
cluster that supports MPI. On your cluster, you need following modules available
to support the experiments.

* Anaconda 3
* Python 3.7+
* Pytorch 1.8+
* Torchvision 0.9+
* Numpy 1.19+
* Matplotlib 3.3+
* Scipy 1.4+
* MPI for Python 3.0+

# Pre-processed data
The *data* directory contains the pre-processed data for the experiments. You can download the data in *.mat* files from https://drive.google.com/drive/folders/1rWJYkfMavGs1F-H0jykJ5V0fIiwrQdJV?usp=sharing.
You can also generate the pre-processed data from the original datasets.

## For the Opp dataset
1. Download the dataset publicly available at
    http://www.opportunity-project.eu/system/files/Challenge/OpportunityChallengeLabeled.zip.
2. Extract all the *.dat* files into the *data/opp* directory.
3. Uncomment the `# gen_opp("data")` in `utils.py` and run `python3 src/utils.py` (it will take a few minutes).
4. Once it's completed, you should find a generated **opp.mat** file in *data/opp/*.

## For the mHealth dataset
1. Download the dataset publicly available at
https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip.
2. Extract all the *.log* files into the *data/mhealth* directory.
3. Uncomment the `# gen_mhealth("data")` in `utils.py` and run `python3 src/utils.py` (it will take a few minutes).
4. Once it's completed, you should find a generated **mhealth.mat** file in *data/mhealth/*.

## For the UR Fall dataset
1. Uncomment the `# download_UR_fall()` and `# gen_ur_fall("data")` in the `utils.py`.
2. Run `python3 src/utils.py` to download the raw data and generate the pre-processed data (it will take a few minutes).
3. Once it's completed, you should find a generated **ur_fall.mat** file in *data/mhealth/*.

# Configuration files
The *config/* directory contains the configuration files of experiments. Each
file describes the parameters of one individual experiment. The
**config/config_example** file contains the explanations of all the parameters.

# Job submission files
The *sub/* directory contains the job submission files to HPC clusters. Each
file uses `mpirun` to run 64 replicates of simulation using a specific
configuration file. You may need to adjust the walltime, number of nodes and
CPUs, and size of memory according to the policy of your clusters (i.e., the
first 3 lines). It uses `module load` to load an Anaconda3 module named as
"anaconda3/personal" and an MPI module named as "mpi" for simulations. Change
them to suit your cluster's environment, if necessary. The \$PBS_O_WORKDIR\$ is
the absolute path of the current working directory of the `qsub` utility
process.

# Instructions
1. Create a conda environment (with Python 3.7+) named as **deep-learning** on
your HPC cluster.
2. Install pytorch, torchvision, numpy, matplotlib, scipy, and mpi4py in your
**deep-learning** environment.
3. Use the provided *.sh* files to run groups of experiments. For
example, the exp_opp.sh will use `qsub` to submit all the pbs files of the Opp
experiments to the queue system of your cluster.
4. With the provided configuration files, the results will be output in the
*results/* directory.
5. Once ALL experiments are completed, run `python3 src/analysis.py` to output
data visualizations of the results into the *plots/* directory.
