"""
Graph Neural Network implementation based on torch_geometric

Implements the Encode - Process - Decode - Material GNN.

Two getStiffness functions are implemented: one using automatic differentiation and one using Finite differences.
FE2_steps functions are re-implementations of the forward functions that closer align with how the GNN would be used in inference. These are used for time comparisons

"""

import time
import torch
from torch import Tensor, cat
from torch.nn import Sequential as Seq, LeakyReLU, ReLU, Sigmoid, Tanh, SiLU, SELU, Dropout
from torch_geometric.nn import MessagePassing, LayerNorm, GraphNorm
from torch_geometric.nn.dense.linear import Linear

from J2Model import J2Material, NanValueError, MaxIterError, NegativeDgamError



class GNN(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, num_layers, settings, device=-1, strainNorm = None, elemepspeqNorm=None, homStressNorm=None, elemStressNorm=None):
        super(GNN, self).__init__()
        self.device = device
        self.matcomputetime = 0

        self.material = J2Material(self.device)

        self.settings = settings

        # Decide whether to run material models.
        self.settings['run_material'] = True
        if self.settings['stress_as_input_field'] or self.settings['fullfield_stress']:
            self.settings['run_material'] = False

        # Stress as input field -> stress to stress
        # fullfield_stress -> strain to strain & stress

        hid_n = self.settings['hidden_nodes']
        w_init = self.settings['w_init']
        bias = self.settings['bias']
        self.res_lay = self.settings['res_lay']
        self.tot_res_lay = self.settings['tot_res_lay']
        self.layers = num_layers
        self.aggr = self.settings['aggregation']

        self.strainNorm = strainNorm
        self.elemepspeqNorm = elemepspeqNorm
        self.homStressNorm = homStressNorm
        self.elemStressNorm = elemStressNorm

        activations = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'leakyrelu': LeakyReLU(),
            'silu': SiLU(),
            'selu': SELU()
        }
        self.act_func = activations[self.settings['activation_func']]

        # Initializing MLPs
        enc_1 = Linear(in_channels, int((in_channels + hid_n) / 2), weight_initializer = w_init, bias=bias)
        enc_2 = Linear(int((in_channels + hid_n)/2), hid_n, weight_initializer = w_init, bias=bias)

        dec_1 = Linear(hid_n, int(hid_n / 2), weight_initializer = w_init, bias=bias)
        dec_2 = Linear(int(hid_n/2), out_channels, weight_initializer = w_init, bias=bias)

        nodeMLP_0_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_0_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_1_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_1_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_2_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_2_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_3_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_3_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_4_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_4_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_5_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_5_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        nodeMLP_6_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
        nodeMLP_6_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        if num_layers > 7:
            nodeMLP_7_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
            nodeMLP_7_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

            nodeMLP_8_1 = Linear(2*hid_n, hid_n, weight_initializer = w_init, bias=bias)
            nodeMLP_8_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        if settings['edge_augmentation'] == 'None':
            additional_inputs = 2
        else:   # one-hot feature is added in data for the augmented edges
            additional_inputs = 3

        edgeMLP_0_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer = w_init, bias=bias)
        edgeMLP_0_2 = Linear(hid_n, hid_n, weight_initializer = w_init, bias=bias)

        edgeMLP_1_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_1_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        edgeMLP_2_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_2_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        edgeMLP_3_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_3_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        edgeMLP_4_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_4_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        edgeMLP_5_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_5_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        edgeMLP_6_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
        edgeMLP_6_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        if num_layers > 7:
            edgeMLP_7_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
            edgeMLP_7_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

            edgeMLP_8_1 = Linear(hid_n + additional_inputs, hid_n, weight_initializer=w_init, bias=bias)
            edgeMLP_8_2 = Linear(hid_n, hid_n, weight_initializer=w_init, bias=bias)

        # Initializing dropout layers, one for after self.encoder, the others for after the nodeMLPs
        self.drop_enc = Dropout(p=settings['dropout'])
        self.drop_0 = Dropout(p=settings['dropout'])
        self.drop_1 = Dropout(p=settings['dropout'])
        self.drop_2 = Dropout(p=settings['dropout'])
        self.drop_3 = Dropout(p=settings['dropout'])
        self.drop_4 = Dropout(p=settings['dropout'])
        self.drop_5 = Dropout(p=settings['dropout'])
        self.drop_6 = Dropout(p=settings['dropout'])
        if num_layers > 7:
            self.drop_7 = Dropout(p=settings['dropout'])
            self.drop_8 = Dropout(p=settings['dropout'])

        # Initializing norm layers
        if self.settings['normlayers'] == 'Graph':
            DecNorm = GraphNorm( hid_n )
            nodeNorm_0 = GraphNorm( 2*hid_n )
            nodeNorm_1 = GraphNorm( 2*hid_n )
            nodeNorm_2 = GraphNorm( 2*hid_n )
            nodeNorm_3 = GraphNorm( 2*hid_n )
            nodeNorm_4 = GraphNorm( 2*hid_n )
        elif self.settings['normlayers'] == 'Layer':
            DecNorm = LayerNorm( hid_n )
            nodeNorm_0 = LayerNorm( 2*hid_n )
            nodeNorm_1 = LayerNorm( 2*hid_n )
            nodeNorm_2 = LayerNorm( 2*hid_n )
            nodeNorm_3 = LayerNorm( 2*hid_n )
            nodeNorm_4 = LayerNorm( 2*hid_n )

        # Combining layers into MLPs
        self.encoder = Seq( enc_1, self.act_func, enc_2, self.act_func, self.drop_enc )


        # No act_func at end because with residual layer -> state can only increase!
        if self.settings['normlayers'] == 'Layer' or self.settings['normlayers'] == 'Graph':
            self.decoder = Seq( DecNorm, dec_1, self.act_func, dec_2 )
            self.nodeupdate_0 = Seq( nodeNorm_0, nodeMLP_0_1, self.act_func, nodeMLP_0_2)
            self.nodeupdate_1 = Seq( nodeNorm_1, nodeMLP_1_1, self.act_func, nodeMLP_1_2)
            self.nodeupdate_2 = Seq( nodeNorm_2, nodeMLP_2_1, self.act_func, nodeMLP_2_2)
            self.nodeupdate_3 = Seq( nodeNorm_3, nodeMLP_3_1, self.act_func, nodeMLP_3_2)
            self.nodeupdate_4 = Seq( nodeNorm_4, nodeMLP_4_1, self.act_func, nodeMLP_4_2)

        else:
            self.decoder = Seq( dec_1, self.act_func, dec_2 )
            self.nodeupdate_0 = Seq( nodeMLP_0_1, self.act_func, self.drop_0, nodeMLP_0_2)
            self.nodeupdate_1 = Seq( nodeMLP_1_1, self.act_func, self.drop_1, nodeMLP_1_2)
            self.nodeupdate_2 = Seq( nodeMLP_2_1, self.act_func, self.drop_2, nodeMLP_2_2)
            self.nodeupdate_3 = Seq( nodeMLP_3_1, self.act_func, self.drop_3, nodeMLP_3_2)
            self.nodeupdate_4 = Seq( nodeMLP_4_1, self.act_func, self.drop_4, nodeMLP_4_2)
            self.nodeupdate_5 = Seq( nodeMLP_5_1, self.act_func, self.drop_5, nodeMLP_5_2)
            self.nodeupdate_6 = Seq( nodeMLP_6_1, self.act_func, self.drop_6, nodeMLP_6_2)
            if num_layers > 7:
                self.nodeupdate_7 = Seq( nodeMLP_7_1, self.act_func, self.drop_7, nodeMLP_7_2)
                self.nodeupdate_8 = Seq( nodeMLP_8_1, self.act_func, self.drop_8, nodeMLP_8_2)


        self.edgeupdate_0 = Seq( edgeMLP_0_1, self.act_func, edgeMLP_0_2 )
        self.edgeupdate_1 = Seq( edgeMLP_1_1, self.act_func, edgeMLP_1_2 )
        self.edgeupdate_2 = Seq( edgeMLP_2_1, self.act_func, edgeMLP_2_2 )
        self.edgeupdate_3 = Seq( edgeMLP_3_1, self.act_func, edgeMLP_3_2 )
        self.edgeupdate_4 = Seq( edgeMLP_4_1, self.act_func, edgeMLP_4_2 )
        self.edgeupdate_5 = Seq( edgeMLP_5_1, self.act_func, edgeMLP_5_2 )
        self.edgeupdate_6 = Seq( edgeMLP_6_1, self.act_func, edgeMLP_6_2 )
        if num_layers > 7:
            self.edgeupdate_7 = Seq( edgeMLP_7_1, self.act_func, edgeMLP_7_2 )
            self.edgeupdate_8 = Seq( edgeMLP_8_1, self.act_func, edgeMLP_8_2 )

        self.node_funcs = [self.nodeupdate_0, self.nodeupdate_1, self.nodeupdate_2, self.nodeupdate_3, self.nodeupdate_4, self.nodeupdate_5, self.nodeupdate_6 ]
        self.edge_funcs = [self.edgeupdate_0, self.edgeupdate_1, self.edgeupdate_2, self.edgeupdate_3, self.edgeupdate_4, self.edgeupdate_5, self.edgeupdate_6 ]
        self.drop_funcs = [self.drop_0, self.drop_1, self.drop_2, self.drop_3, self.drop_4, self.drop_5, self.drop_6 ]
        if num_layers > 7:
            self.node_funcs.append(self.nodeupdate_7)
            self.node_funcs.append(self.nodeupdate_8)
            self.edge_funcs.append(self.edgeupdate_7)
            self.edge_funcs.append(self.edgeupdate_8)
            self.drop_funcs.append(self.drop_7)
            self.drop_funcs.append(self.drop_8)

    def forward(self, data, timesteps=None, noplasticity=False):
        """
        Description outdated (last updated 11-2022)
        datasets.append(Data(x=combined_inputs, macro=macro_strain_after, epsp=epsp_input, edge_index=edges.t().contiguous().to(device), edge_feat=edge_feats, y=targets))
        data {
        x = [eps_micro0, eps_macro0, epsp_eq0, fib_feats]       (N, 13)
        macro = [macro1, macro2, macro..]                       (N, t, 3)
        epsp = [epsp0]                                          (N, 4)
        edge_index = [edges]                                    (2, E)
        edge_feat = [edge_feats]                                (E) OR Vector (E, 2)
        noplasticity -> Plasticity is not allowed. (a limit is set, as even predicting for 0 inputs often leads to some elements being plastic...)
        }
        """
        if timesteps is None:
            timesteps = data.t_pred[0].item()

        num_nodes = data.x.shape[0]

        pred_strains = torch.empty((num_nodes, timesteps, 3)).to(self.device)

        # Initialize material model
        if self.settings['run_material']:
            self.material.configure(num_nodes)

        pred_stress = torch.empty_like(pred_strains)
        pred_eps_p_eqs = torch.empty((num_nodes, timesteps)).to(self.device)

        num_batches = len(data.num_voids)  # Each batch has a single value for number of voids. The length therefore gives us the number of batches.

        newdata = data.x.clone()

        for t in range(timesteps):
            # Preprocess
            """
            Encode
            """
            state = self.encoder(newdata)

            if self.tot_res_lay:
                start_state = state  # save initial state for residual

            """
            Process
            """
            for layer in range(self.layers):

                if self.settings['unique_MPLs']:
                    MPL_num = layer
                else:
                    MPL_num = min(layer, 2)     # Share weights for every layer after the first 2. (Not used in paper)

                # Messages
                h_v_all = self.propagate(data.edge_index, x=state, edge_feat=data.edge_feat, MPL_num= MPL_num)

                # Update nodal states based on current state & messages
                res = self.node_funcs[MPL_num]( cat((state, h_v_all), 1) )

                # Residual layer
                if self.res_lay:
                    state = state + res
                else:
                    state = res

            if self.tot_res_lay: # Residual layer total
                state = state + start_state
            """
            Decode
            """
            pred = self.decoder(state)        # Micro strain (and maybe stress) prediction

            # Store output strains (default scenario) for material model and returning
            if self.settings['fullfield_stress']:
                eps_new = pred[:,:3]
            else:
                eps_new = pred

            if timesteps > 1:
                pred_strains[:, t] = eps_new
            else:
                pred_strains = eps_new

            """
            Material model
            """
            unnorm_strains = self.strainNorm.denormalize(eps_new)

            if self.settings['run_material']:
                # Use material model
                try:
                    pred_stress[:, t] = self.material.update(unnorm_strains)
                except NanValueError:
                    print("dgam contained nan value, aborted findRoot")
                    self.settings['skip_minibatch'] = True
                    break
                except MaxIterError:
                    print("Maximum number of iterations reached, aborted findRoot")
                    self.settings['skip_minibatch'] = True
                    break
                except NegativeDgamError:
                    print("ddgam contained negative values, aborted findRoot")
                    self.settings['skip_minibatch'] = True
                    break
                except:
                    print("Update material model failed due to unknown error.")
                    self.settings['skip_minibatch'] = True
                    self.settings['abort_training'] = True
                    break
            # Don't use material model.
            elif self.settings['stress_as_input_field']:
                # In case of stress as input, unnorm_strains are actually stresses
                pred_stress[:, t] = unnorm_strains
            elif self.settings['fullfield_stress']: # Output is directly both strains and stresses.
                pred_stress[:, t] = self.elemStressNorm.denormalize(pred[:, 3:6])     # Still normalized

            if t + 1 != timesteps:
                newdata = data.x.clone()

                # Prepare data for next timestep
                newdata[:,3:6] = data.macro[:, t]   # add macro of next step
                newdata[:,0:3] = eps_new            # add previous predicted micro (strain or stress)
                if self.settings['epspeq_feature']:
                    newdata[:, 6] = self.elemepspeqNorm.normalize(self.material.new_epspeq_hist)
            if noplasticity:
                sum_epspeq = torch.sum(self.material.new_epspeq_hist)
                assert sum_epspeq < 0.1, f'Plasticity is not allowed! It is {sum_epspeq} but should be 0.'

            if self.settings['run_material']:
                # Store epsp for plotting
                pred_eps_p_eqs[:, t] = self.material.new_epspeq_hist
                # Commit history
                self.material.commit()

        # Compute homogenized stress for all timesteps
        if self.settings['run_material'] or self.settings['stress_as_input_field'] or self.settings['fullfield_stress']:
            hom_stress = self.computeHomStress(data, pred_stress, num_batches, timesteps)
            hom_stress = self.homStressNorm.normalize(hom_stress)
        else:
            hom_stress = None

        if len(pred_strains.shape) != 3:
            pred_strains = pred_strains.view(-1, timesteps, 3)

        return pred_strains, hom_stress, pred_stress, pred_eps_p_eqs
               #strains normalized, hom_stress normalized, pred_stress unnormalized, pred_eps_p_eqs unnormalized


    def message(self, x_i: Tensor, x_j: Tensor, edge_feat, MPL_num) -> Tensor:
        return self.edge_funcs[MPL_num](cat([x_j, edge_feat], dim=1))


    def computeHomStress(self, data, pred_stresses, num_batches, timesteps):
        """
        Computes homogenized stresses for a batch of meshes simultaniously
        NOTE: Assumes computation on undeformed analytical area, not the mesh
        data: data object form dataloader
        pred_stresses: predicted stresses for all nodes in batches all timesteps
        num_batches: number of batches
        """
        ipArea = data.ip_area
        ids = data.id
        num_voids = data.num_voids      # e.g. [3, 3, 3, 49]

        # We use unique_ids to track which elements in the batch belong to which sample.
        unique_ids = torch.unique(ids)

        hom_stresses = torch.empty(( num_batches, timesteps, 3), device=self.device)

        # Compute total area for each batch
        areas = num_voids * torch.pi * self.settings['r_fiber']**2 / self.settings['vfrac']

        for t in range(timesteps):
            area_weighted_stress = (pred_stresses[:, t].T * ipArea).T

            for i, (id, area) in enumerate( zip(unique_ids, areas) ):
                # Select subset of stress for each sample in the batch, sum over each component and divide by area.
                hom_stresses[i, t] = torch.sum(area_weighted_stress[ids == int(id)], axis=0) / area

        return hom_stresses

    def getStiffness(self, data, macroStrainNorm, strain_in):
        """
        Autograd
        """
        stiffness = torch.zeros((3,3))

        # Input strain to track gradients
        strain_in = strain_in.clone().detach().requires_grad_(True)
        strain_norm = macroStrainNorm.normalize(strain_in)

        # Insert macro strain into data
        data.x[:, 3:6] = torch.broadcast_to(strain_norm, data.x[:, 3:6].shape)

        # Run model for a single step
        _, hom_stress, _, _ = self.forward(data, timesteps=1, noplasticity=True)       # returns normalized stress

        hom_stress = self.homStressNorm.denormalize(hom_stress)

        for i in range(3):
            stiffness[i] = torch.autograd.grad(hom_stress[0,0,i], strain_in , create_graph=True)[0].detach()

        # # stiffness = torch.autograd.grad(hom_stress[0, 0, :], strain_in, torch.eye(3),is_grads_batched=True)[0]

        return stiffness

    def getStiffnessFD(self, data, macroStrainNorm, eps=1e-5):
        """
        Finite Differences
        """
        stiffness = torch.zeros((3,3))

        # Add a normalized 0 prediction
        value = 0.0
        strain_fd = torch.tensor([value, value, value])

        for i in range(3):
            # Forward
            strain_fd[i] += eps
            data.x[:, 3:6] = torch.broadcast_to(macroStrainNorm.normalize(strain_fd), data.x[:, 3:6].shape)
            _, fwd_hom_stress, _, _ = self.forward(data, timesteps=1)

            # Backward
            strain_fd[i] -= 2 * eps
            data.x[:, 3:6] = torch.broadcast_to(macroStrainNorm.normalize(strain_fd), data.x[:, 3:6].shape)
            _, bck_hom_stress, _, _ = self.forward(data, timesteps=1)

            strain_fd[i] += eps # reset strain_fd

            # (fwd-bck)/(2*eps)
            stiffness[:,i] = ( self.homStressNorm.denormalize(fwd_hom_stress) - self.homStressNorm.denormalize(bck_hom_stress)) / (2 * eps)  # * self.homStressNorm.factor

        return stiffness


    def FE2_steps(self, data, timesteps=None):

        # Preprocess
        with (torch.no_grad()):
            if timesteps is None:
                timesteps = data.t_pred[0].item()

            # Reset the material model parameters
            num_nodes = data.x.shape[0]
            if self.settings['run_material']:
                self.material.configure(num_nodes)

            newdata = data.x.clone()
            stiffnesses = torch.empty((timesteps, 3, 3))
            hom_stresses = torch.empty((timesteps, 3))

            next_macro = newdata[0,3:6]

        for t in range(timesteps):
            # Get macro strain for each batch, track gradients (starts in normalized form already)
            macro_tensor = next_macro.clone().detach().requires_grad_(True)
            newdata[:, 3:6] = torch.broadcast_to(macro_tensor, (num_nodes, 3))

            """
            Encode
            """
            state = self.encoder(newdata)

            if self.tot_res_lay:
                start_state = state  # save init for residual

            """
            Process
            """
            for layer in range(self.layers):

                if self.settings['unique_MPLs']:
                    MPL_num = layer
                else:
                    MPL_num = min(layer, 2)  # Share weights for every layer after the first 2

                h_v_all = self.propagate(data.edge_index, x=state, edge_feat=data.edge_feat, MPL_num=MPL_num)

                # forward next MPL
                res = self.node_funcs[MPL_num](cat((state, h_v_all), 1))

                # Residual layer
                if self.res_lay:
                    state = state + res
                else:
                    state = res

            if self.tot_res_lay:  # Residual layer total
                state = state + start_state
            """
            Decode
            """
            pred = self.decoder(state)  # Micro strain (and maybe stress) prediction

            # Store output strains (default scenario) for material model and returning
            if self.settings['fullfield_stress']:
                eps_new = pred[:, :3]
            else:
                eps_new = pred

            """
            Material model
            """
            unnorm_strains = self.strainNorm.denormalize(eps_new)

            mat_start = time.time()

            if self.settings['run_material']:
                cur_stress = self.material.update(unnorm_strains)
            elif self.settings['stress_as_input_field']:
                cur_stress = unnorm_strains
            elif self.settings['fullfield_stress']:  # Output is directly both strains and stresses.
                cur_stress = self.elemStressNorm.denormalize(pred[:, 3:6])  # Still normalized

            self.matcomputetime += time.time() - mat_start

            """
            Compute homogenized inside time loop
            """
            # Compute homogenized stress single timestep
            hom_stress = self.computeHomStress(data, cur_stress.view(-1,1,3), 1, 1)  # only single step, single batch

            """
            Compute stiffness
            """

            for i in range(3):
                if i == 2: # Not having "create_graph" for the final operation is crucial to prevent cuda memory from exploding.
                    stiffnesses[t,i] = torch.autograd.grad(hom_stress[0, 0, i], macro_tensor)[0].detach()
                else:
                    stiffnesses[t,i] = torch.autograd.grad(hom_stress[0, 0, i], macro_tensor, retain_graph=True)[0].detach()

            # Store homstress
            hom_stresses[t] = hom_stress.detach()

            """
            Data processing for next step
            """
            with torch.no_grad():
                if t + 1 != timesteps:

                    # Prepare data for next timestep
                    next_macro = data.macro[0, t]
                    newdata[:, 0:3] = eps_new.detach()  # add previous predicted micro (strain or stress)
                    if self.settings['epspeq_feature']:
                        self.material.new_epspeq_hist = self.material.new_epspeq_hist.detach()
                        newdata[:, 6] = self.elemepspeqNorm.normalize(self.material.new_epspeq_hist)

                    if self.settings['run_material']:   # can do this inside t+1 if since I don't store values.
                        # Commit history, make sure gradients are removed
                        self.material.new_epsp_hist = self.material.new_epsp_hist.detach()
                        self.material.commit()

            torch.cuda.empty_cache()

        return hom_stresses, stiffnesses

    def FE2_steps_nostiff(self, data, timesteps=None):

        # Preprocess
        with (torch.no_grad()):
            if timesteps is None:
                timesteps = data.t_pred[0].item()

            # Reset the material model parameters
            num_nodes = data.x.shape[0]
            if self.settings['run_material']:
                self.material.configure(num_nodes)

            newdata = data.x.clone()
            hom_stresses = torch.empty((timesteps, 3))

            for t in range(timesteps):
                """
                Encode
                """
                state = self.encoder(newdata)

                if self.tot_res_lay:
                    start_state = state  # save init for residual

                """
                Process
                """
                for layer in range(self.layers):

                    if self.settings['unique_MPLs']:
                        MPL_num = layer
                    else:
                        MPL_num = min(layer, 2)  # Share weights for every layer after the first 2

                    h_v_all = self.propagate(data.edge_index, x=state, edge_feat=data.edge_feat, MPL_num=MPL_num)

                    # forward next MPL
                    res = self.node_funcs[MPL_num](cat((state, h_v_all), 1))

                    # Residual layer
                    if self.res_lay:
                        state = state + res
                    else:
                        state = res

                if self.tot_res_lay:  # Residual layer total
                    state = state + start_state
                """
                Decode
                """
                pred = self.decoder(state)  # Micro strain (and maybe stress) prediction

                # Store output strains (default scenario) for material model and returning
                if self.settings['fullfield_stress']:
                    eps_new = pred[:, :3]
                else:
                    eps_new = pred

                """
                Material model
                """
                unnorm_strains = self.strainNorm.denormalize(eps_new)

                mat_start = time.time()

                if self.settings['run_material']:
                    cur_stress = self.material.update(unnorm_strains)
                elif self.settings['stress_as_input_field']:
                    cur_stress = unnorm_strains
                elif self.settings['fullfield_stress']:  # Output is directly both strains and stresses.
                    cur_stress = self.elemStressNorm.denormalize(pred[:, 3:6])  # Still normalized

                self.matcomputetime += time.time() - mat_start

                """
                Compute homogenized inside time loop
                """
                # Compute homogenized stress single timestep
                hom_stresses[t] = self.computeHomStress(data, cur_stress.view(-1, 1, 3), 1, 1)  # only single step, single batch

                """
                Data processing for next step
                """
                if t + 1 != timesteps:
                    # Prepare data for next timestep
                    newdata[:, 3:6] = data.macro[:, t]
                    newdata[:, 0:3] = eps_new  # add previous predicted micro (strain or stress)
                    if self.settings['epspeq_feature']:
                        newdata[:, 6] = self.elemepspeqNorm.normalize(self.material.new_epspeq_hist)

                    if self.settings['run_material']:  # can do this inside t+1 if since I don't store values.
                        # Commit history, make sure gradients are removed
                        self.material.commit()

        return hom_stresses

