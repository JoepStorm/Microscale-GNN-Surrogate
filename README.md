# Microscale GNN Surrogate
This is a repository with the code corresponding to the paper "[A Microstructure-based Graph Neural Network for Accelerating Multiscale Simulations](https://www.sciencedirect.com/science/article/pii/S0045782524002573)".

## Full-field microscale predictions:

<p float="left">
  <img src="https://github.com/JoepStorm/Microscale-GNN-Surrogate/blob/main/results/git_anim_stress_pred.gif" width="750" />
  <img src="https://github.com/JoepStorm/Microscale-GNN-Surrogate/blob/main/results/git_anim_stress_pred_hom.gif" width="250" /> 
</p>


## Extrapolation to larger microstructures:

<p float="left">
  <img src="https://github.com/JoepStorm/Microscale-GNN-Surrogate/blob/main/results/git_anim_stress_extrap.gif" width="750" />
  <img src="https://github.com/JoepStorm/Microscale-GNN-Surrogate/blob/main/results/git_anim_stress_extrap_hom.gif" width="250" /> 
</p>

## Network architecture overview:
<img src="https://github.com/JoepStorm/Microscale-GNN-Surrogate/blob/main/Network_architecture.png" width="750" /> 

# Using these files:

[Accompanying data files can be found at the 4TU repository](https://data.4tu.nl/datasets/f2a20379-0d48-4829-a5a2-c080eb669663) and should be put in the main folder.
The script to create the datasets is not provided, it uses an in-house modified version of the Jem/Jive library.
Additional datasets were used to create the paper, and can be requested.

Inference using existing models:
Use load_model.py, and select the experiment configuration.
The configuration reads additional settings in the corresponding "experiments/.../exp_config" file.
We've included pre-trained GNN models which can be directly used. 
Full-field plots are created using load_model.py by specifying the mesh id or by letting it print representative samples.

Training new GNNs:
Set-up the configuration settings in "fem/ConfigGNN" file, or programmatically in "readConfig.py" (this will overwrite any ConfigGNN setting).
Run "python trainGNN.py 1" (The integer variable is required to read the correct settings).
Results will be stored in the results folder.

Plotting:
The following scripts can be used and adjusted to recreate all figures.
- plotMainResults.py
- plotStiff.py
- plotBar.py
- dogbone/plot_timeComp.py

Any questions related to the usage of this code can be sent to j.storm@tudelft.nl
