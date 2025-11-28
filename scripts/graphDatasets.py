"""
Script that matches the dataset with the created graph to create Data samples ready for PyTorch DataLoaders
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data
from createGraph import volumeConsistentGraph
import normUtils


def createDataset(settings, init_tensor_dtype, macro_vec_train, homstress_train, macro_vec_transfer, homstress_transfer, fibers, trainfibnum, strain_components, pred_components, elemStrainNorm, elemepspeqNorm, eps_set, epspeq_set_noise, geom_features, normalized_stresses, normclassexists, geomFeatNorm, edgeFeatNorm):
    mesh_elem_nodes_set = []
    mesh_node_coords_set = []
    mesh_ipArea_set = []
    datasets = []

    for m in range( settings['transferend'] ):
        if m < settings['validationend']:
            meshfile = f"{settings['meshdir']}/m_{m}.msh"
            macro_vec = macro_vec_train
            homstress = homstress_train
            index_macro_vec = m
            steps_ahead = settings['steps_ahead']
            try:
                settings['vfrac'] = settings['vfrac_train']
            except:
                print(f"vfrac setting not found in settings dictionary. This is okay for non-vfrac experiments.")
        else:
            meshfile = f"{settings['transfermesh']}/m_{m-settings['validationend']}.msh"
            macro_vec = macro_vec_transfer
            homstress = homstress_transfer
            index_macro_vec = m - settings['validationend']
            steps_ahead = settings['transfersteps']
            try:
                settings['vfrac'] = settings['vfrac_trans']
            except:
                print(f"vfrac setting not found in settings dictionary. This is okay for non-vfrac experiments.")

        if m % 100 == 0:
            print(f"Creating graph {m} from {meshfile}")

        ### Get fiber coordinates ###
        curfibers = np.array(fibers[m])
        fiberpositions = curfibers[(np.arange(curfibers.shape[0]) % (3)) < 2].reshape(-1, 2).T  # Extract x and y, without r

        # Added to dataset for homogenization
        num_voids = len(curfibers) / 3

        ### Create mesh ###
        graph_edges, edge_feats, fib_feats, num_nodes, mesh_elem_nodes, mesh_node_coords, ipArea = volumeConsistentGraph(meshfile, settings, fiberpositions)

        ipArea = torch.tensor(ipArea)

        mesh_elem_nodes_set.append(mesh_elem_nodes)
        mesh_node_coords_set.append(mesh_node_coords)
        mesh_ipArea_set.append(ipArea)

        ### Normalize graph features ###
        edges = torch.tensor(graph_edges) - 1
        edge_feats = torch.tensor(edge_feats, dtype=init_tensor_dtype)

        normal_edges = edge_feats.shape[0]
        if settings['edge_augmentation'] == 'NoFeats':
            # If edge_augmentations without distance features are used ([0,0,1]), exclude them from normalizing
            # Augmented edges with [dx, dy] are included in the normalization, even if their dx & dy are much larger.
            # Edges are always added in pairs, therefore the ' /2 -> floor *2 ' is needed to get the correct value of original edges
            normal_edges = math.floor(normal_edges / (1+settings['augmentation_percentage']) / 2) * 2

        if trainfibnum > 0:
            # Initialize normalization class for all fiber geometric features NOTE: input features are only normalized based on scaling of the first mesh
            if not normclassexists:
                geomFeatNorm = normUtils.normUnitvar(fib_feats, donorm=settings['norm_inputs'], normmean=0)   # specify normmean to keep centered around 0.
                if settings['edge_vec']:
                    edgeFeatNorm = normUtils.normUnitvar(edge_feats[:normal_edges,:2], donorm=settings['norm_inputs'], normmean=0)
                normclassexists = True
            fib_feats = geomFeatNorm.normalize(fib_feats)
            edge_feats[:normal_edges,:2] = edgeFeatNorm.normalize(edge_feats[:normal_edges,:2])

            fib_feats = np.transpose(np.tile(fib_feats, 1))

        # Fiber features to torch tensor
        fib_feats = torch.tensor(fib_feats, dtype=init_tensor_dtype).view(-1, num_nodes, geom_features)

        """
        Creating a torch_geometric Data class
        """
        size_straininputs = strain_components * num_nodes

        ### Create 0 start tensor for first prediction step ###
        strain_inputs = elemStrainNorm.normalize(torch.zeros((num_nodes, strain_components), dtype=init_tensor_dtype))
        if settings['epspeq_feature']:
            epsp_eq_input = elemepspeqNorm.normalize(torch.zeros((num_nodes, 1), dtype=init_tensor_dtype))

        macro_strain_direct = torch.tensor(np.broadcast_to(np.array(macro_vec[index_macro_vec, 0]), (num_nodes, 3)), dtype=init_tensor_dtype).view(num_nodes, 3)  # Reshape macro strain
        macro_strain_after = torch.tensor(np.broadcast_to(np.array(macro_vec[index_macro_vec, 1:steps_ahead]), (num_nodes, steps_ahead-1, 3)))

        ### Create the target vector ###
        idx = np.arange(num_nodes * steps_ahead).reshape(steps_ahead, -1).T
        targets = torch.tensor(np.array(eps_set[m][0:steps_ahead * size_straininputs]), dtype=init_tensor_dtype).view(-1, pred_components)[idx]

        stress_targets = torch.tensor(np.array(normalized_stresses[m][0:steps_ahead * size_straininputs]), dtype=init_tensor_dtype).view(-1, pred_components)[idx]

        # Create the homogenized stress target vector
        homstress_target = torch.tensor( homstress[index_macro_vec], dtype=init_tensor_dtype).view(1, -1, 3)

        if settings['epspeq_feature']:
            combined_inputs = torch.cat((strain_inputs, macro_strain_direct, epsp_eq_input, fib_feats[0]), 1)
            epsp_eq_after = torch.tensor(np.array(epspeq_set_noise[m][0:(steps_ahead - 1) * num_nodes]), dtype=init_tensor_dtype)[idx[:, :steps_ahead - 1]]
        else:
            combined_inputs = torch.cat((strain_inputs, macro_strain_direct, fib_feats[0]), 1)
            epsp_eq_after = None
        if settings['numvoid_feature']:# Create a feature for the number of voids
            num_void_value = np.sqrt(num_voids) / 3   # Somewhat arbitrary. This makes 9 -> 1, 81 -> 3
            num_voids_features = torch.tensor(np.broadcast_to(np.array(num_void_value), (num_nodes, 1)), dtype=init_tensor_dtype).view(num_nodes, 1)

            combined_inputs = torch.cat((combined_inputs, num_voids_features), 1)

        ### Create the datasets ###
        id = torch.ones(num_nodes) * m  # add id for homogenizing: tracking which node belongs to which sample in a batch

        datasets.append( Data(x=combined_inputs, macro=macro_strain_after, edge_index=edges.t().contiguous(), edge_feat=edge_feats, y=targets, y_stress=stress_targets, t_pred=steps_ahead, target_homstress=homstress_target, ip_area=ipArea, id=id, num_voids = num_voids, epspeq_after=epsp_eq_after))
    return datasets, mesh_elem_nodes_set, mesh_node_coords_set, mesh_ipArea_set, geomFeatNorm, edgeFeatNorm
