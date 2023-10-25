# GRELinker (A Graph-based Generative Model for Molecular Linker Design with Reinforcement and Curriculum Learning)

This is the code for the "GRELinker: A Graph-based Generative Model for Molecular Linker Design with Reinforcement and Curriculum Learning".

## Prerequisites

* Anaconda or Miniconda with Python 3.6 or 3.8.

* CUDA-enabled GPU.

## Install requirements

Create a new conda environment:

```bash
conda env create -f environment.yml
conda activate GRELinker-env
```

## Pre-processing 

We use the same datasets as [SyntaLinker](https://github.com/YuYaoYang2333/SyntaLinker), the data was originated from the ChEMBL database in the `data/` folder.

For Input file generation, we run the `submitPT.py` script, and the job_type can be set to "preprocess".

## Pre-training

The data has already been preprocessed for training the GRELinker model.

Model training can be started by running the `submitPT.py` script, and the job_type can be set to "train".

The model can be found in the your job_dir folder.

## Generation

The model can be used while training or after training.

To generate the predictions use the `submitPT.py` script, and the job_type can be set to "generate".

## Reinforcement Learning

If you have the best pretrained-model while training, we can fine-tune the model by running the `submitFT.py` script, and the job_type can be set to "learn".

The score function can be set as one of the score components: "reduce", "augment", "qed", "activity","3D_SMI","docking_score" or "SA".

When using "docking_score" components, you need to set the config file in `DockStream/Glide_demo/Glide_docking.json`.

DockStream [tutorial notebook](https://github.com/MolecularAI/ReinventCommunity/blob/master/notebooks/Reinforcement_Learning_Demo_DockStream.ipynb) is provided.

## Curriculum Learning

If you want to fine-tune the model to generate linkers which are structurally complex, we can run the `submitCL.py` script.

The config file example can be seen in "AutoCL_demo/AutoCL_config.json".

## Tools

The other tools to evaluate metrics, such as RMSD, 3D smiliarity in case study can be found in `Utils/` folder.

## Related work

The code was built based on Reinvent (https://github.com/MolecularAI/Reinvent), DockStream (https://github.com/MolecularAI/DockStream),

GraphINVENT(https://github.com/MolecularAI/GraphINVENT), RL-GraphINVENT(https://github.com/olsson-group/RL-GraphINVENT).

Thanks a lot for their sharing.

## Citation

If you find this work useful in your research, please consider citing the paper:

"GRELinker:A Graph-based Generative Model for Molecular Linker Design with Reinforcement and Curriculum Learning".
