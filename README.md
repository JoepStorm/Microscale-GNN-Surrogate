# Microscale GNN Surrogate
This is a repository with the code corresponding to the paper "[A Microstructure-based Graph Neural Network for Accelerating Multiscale Simulations](https://arxiv.org/abs/2402.13101)"

[Accompanying data files can be found at the 4TU repository](https://data.4tu.nl/datasets/f2a20379-0d48-4829-a5a2-c080eb669663) and should be put in the main folder.
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
Full-field plots are created using load_model.py by specifying the mesh id or by letting it print representative samples.

Plotting:
The following scripts can be used and adjusted to recreate all figures.
- plotMainResults.py
- plotStiff.py
- plotBar.py
- dogbone/plot_timeComp.py
