# Microscale GNN Surrogate
This is a repository with the code corresponding to the paper "A Microstructure-based Graph Neural Network for Accelerating Multiscale Simulations"

Paper: <TODO add link>

Accompanying data files can be found at the 4TU repository: <TODO add link> and should be put in the main folder.
The script to create the datasets is not provided, it uses an in-house modified version of the Jem/Jive library.
Additional datasets were used in the creation of the paper, and can be requested.


# Using these files:

Training:
Set-up the configuration settings in "fem/ConfigGNN" file, or program in "readConfig.py".
Run "python trainGNN.py 1". An integer variable is required to read the correct settings.
Results will be stored in the results folder.

Inference:
Use load_model.py, and select the experiment configuration.
The configuration reads additional settings in the corresponding "experiments/.../exp_config" file.
We've included pre-trained GNN models which can be directly used.

Plotting:
The following scripts can be used and adjusted to recreate all figures.
- plotMainResults.py
- plotStiff.py
- plotBar.py
