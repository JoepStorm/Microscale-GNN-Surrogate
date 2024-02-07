"""
Read a mesh file, create and store the graph
Edges are stored explicitly, no adjacency matrix is computed.

phantom nodes are used for periodic edges, connecting nodes on opposite sides.
"""

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
Utility functions
"""

def elemCenter(node_list, elements):
    """
    Function that finds the center of each triangular element in a FEM Mesh
    :param node_list: List of coordinates corresponding to each node
    :param elements: Element considered
    :return: An array with [x, y] the center coordinates of the specified element
    """
    node0 = np.array(node_list[elements[0]-1])  # -1 since elements count nodes startint at 1
    node1 = np.array(node_list[elements[1]-1])
    node2 = np.array(node_list[elements[2]-1])
    center = (node0 + node1 + node2) / 3
    # Area: abs ( (x1y2 + x2y3 + x3y1 - y1x2 - y2x3 - y3x1 ) / 2 )
    area = abs ( ( node0[0]*node1[1] + node1[0]*node2[1] + node2[0]*node0[1] - node0[1]*node1[0] - node1[1]*node2[0] - node2[1]*node0[0] ) / 2 )
    return center, area

def phantomPBCPoint(pointA, pointB, rveBounds):
    """
    :param pointA: coordinates of point A ('Real' point)
    :param pointB: coordinates of point B (To be 'Periodic' point)
    :return: phantom coordinates of point B
    """
    dx = pointA[0] - pointB[0]
    dy = pointA[1] - pointB[1]
    meshwidth  = rveBounds[0][1] - rveBounds[0][0]
    meshheight = rveBounds[1][1] - rveBounds[1][0]
    if abs(dx) > abs(dy):  # Horizontal boundary
        if dx > 0:
            phantomPointB = [pointB[0] + meshwidth, pointB[1]]
        else:
            phantomPointB = [pointB[0] - meshwidth, pointB[1]]
    else:   # Vertical boundary
        if dy > 0:
            phantomPointB = [pointB[0], pointB[1] + meshheight]
        else:
            phantomPointB = [pointB[0], pointB[1] - meshheight]
    return phantomPointB

def euclidDist(pointA, pointB, rveBounds, periodic=False):
    """
    :param pointA: coordinates of point A
    :param pointB: coordinates of point B
    :return: euclidean distance (sqrt((A1-A2)**2+(B1-B2)**2)
    """
    if periodic:
        pointB = phantomPBCPoint(pointA, pointB, rveBounds)

    dx = max(pointA[0], pointB[0]) - min(pointA[0], pointB[0])
    dy = max(pointA[1], pointB[1]) - min(pointA[1], pointB[1])
    # Points can share a coordinate (common in denser meshes). Adding a tiny value to prevent division by 0
    if dx == 0:
        dx += 1e-8
    if dy == 0:
        dy += 1e-8

    return 1/np.sqrt(dx**2+dy**2), [pointA[0] - pointB[0], pointA[1] - pointB[1]]

def checkMeshEdge(pointA, pointB, rveBounds):
    # Check if both points have at least one coordinate at a boundary value:
    # print(f"A: {pointA}, B: {pointB}")
         # Right, Top, Left, Bot    # Ugly but saves initializing 8 booleans
    A = [False, False, False, False]
    B = [False, False, False, False]
    EdgeType = "None"
    num_diff = 1e-8

    if abs(pointA[0] - rveBounds[0][1]) < num_diff and abs(pointB[0] - rveBounds[0][1]) < num_diff: # Right
        EdgeType = "Right"
    if abs(pointA[1] - rveBounds[1][1]) < num_diff and abs(pointB[1] - rveBounds[1][1]) < num_diff:  # Top
        EdgeType = "Top"
    if abs(pointA[0] - rveBounds[0][0]) < num_diff and abs(pointB[0] - rveBounds[0][0]) < num_diff: # Left
        EdgeType = "Left"
    if abs(pointA[1] - rveBounds[1][0]) < num_diff and abs(pointB[1] - rveBounds[1][0]) < num_diff: # Bot
        EdgeType = "Bot"

    return EdgeType

def knn_fibdists(fib_x, fib_y, nodeCoords, rveBounds, num_nbrs):
    """
    :param fib_x: Array of x coordinates of the fibers
    :param fib_y: Array of y coordinates of the fibers
    :param nodeCoords: Matrix of node coordinates (Usually Integration Points)
    :Does not work with fibers with varying radii, as only the center point is used
    :return: array: (2*num_nbrs, len(nodeCoords)). [[dx1n1, dx1n2, .. dx1nN] [dy1n1, .. dy1nN] .. [dyFn1, .. dyFnN]]
    """

    if fib_x.shape[0] <= 3:
        periodicities = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [-1, 2], [-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1]]
    else:
        periodicities = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    # ---- Create full list of periodic fibers
    fib_x_complete = np.array([])
    fib_y_complete = np.array([])
    for i in range(len(periodicities)):
        fib_x_complete = np.append(fib_x_complete, fib_x + periodicities[i][0] * (rveBounds[0][1] - rveBounds[0][0]) )
        fib_y_complete = np.append(fib_y_complete, fib_y + periodicities[i][1] * (rveBounds[1][1] - rveBounds[1][0]) )
    fib_combined = np.array([fib_x_complete, fib_y_complete]).T

    # ## Visual check: print original and added points
    # import matplotlib.pyplot as plt
    # plt.scatter(fib_x_complete[0:3], fib_y_complete[0:3], c='red')
    # plt.scatter(fib_x_complete[3:], fib_y_complete[3:], c='green')
    # plt.savefig(f"knn_periodicity", bbox_inches='tight', pad_inches=0.01, dpi=300, format = 'pdf')
    # exit()

    # ---- Compute nearest neighbours: find closest fibers for every element
    if num_nbrs > 0:
        nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='auto').fit(fib_combined)
        distances, indices = nbrs.kneighbors(nodeCoords)
        # ---- Compute dx & dy distances to neighbours
        dist_arr = np.empty((2*num_nbrs, len(nodeCoords)))
        for i in range(len(nodeCoords)):
            for j in range(num_nbrs):
                dist_arr[j*2, i] = fib_combined[indices[i, j], 0] - nodeCoords[i, 0] # dx
                dist_arr[j*2+1, i] = fib_combined[indices[i, j], 1] - nodeCoords[i, 1] # dy
        return dist_arr
    else:
        return 0


def plotGraph(ipCoords, graph_edge, periodicity, PBC, rveBounds):
    # --- Plot graph
    print("plotting")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for i, edge in enumerate(graph_edge):  # [::2]
        # print(f"i: {i}, edge: {edge}")
        if periodicity:
            if i in PBC:
                newP = phantomPBCPoint(ipCoords[edge[0] - 1, :], ipCoords[edge[1] - 1, :], rveBounds)
                plt.plot([ipCoords[edge[0] - 1, 0], newP[0]],
                         [ipCoords[edge[0] - 1, 1], newP[1]], color='black', linewidth=.5, zorder=1)
            else:
                plt.plot([ipCoords[edge[0] - 1, 0], ipCoords[edge[1] - 1, 0]], [ipCoords[edge[0] - 1, 1], ipCoords[edge[1] - 1, 1]], color='black', linewidth=.5, zorder=1)

                # # Unique colors for all augmented edges (Manually set value!)
                # if i < 1652:
                #     plt.plot([ipCoords[edge[0] - 1, 0], ipCoords[edge[1] - 1, 0]],[ipCoords[edge[0] - 1, 1], ipCoords[edge[1] - 1, 1]], color='black', linewidth=.5, zorder=1)
                # else:
                #     plt.plot([ipCoords[edge[0] - 1, 0], ipCoords[edge[1] - 1, 0]], [ipCoords[edge[0] - 1, 1], ipCoords[edge[1] - 1, 1]], c=np.random.rand(3,), linewidth=.5, zorder=3)
        else:
            plt.plot([ipCoords[edge[0]-1, 0], ipCoords[edge[1]-1, 0]], [ipCoords[edge[0]-1, 1], ipCoords[edge[1]-1, 1]] ,color='black', linewidth=1, zorder=1)

    ax.scatter(ipCoords[:, 0], ipCoords[:, 1], color='white', linewidths=.7, edgecolors='black', s=8, zorder=2)

    # Annotations to check specific values
    # for i in range(len(mesh_elem_nodes)):
        # ax.annotate(f"{sep_fib_dists[0, i]:.2f}",    (ipCoords[i, 0] + 0.01, ipCoords[i, 1] + 0.01), fontsize=3)
        # ax.annotate(f"{sep_fib_dists[1, i]:.2f}",    (ipCoords[i, 0] + 0.01, ipCoords[i, 1] + 0.01), fontsize=3)
        # ax.annotate(f"{target_set[m][i]:.2f}",    (ipCoords[i, 0] + 0.01, ipCoords[i, 1] + 0.01), fontsize=3)

    plt.xticks([])
    plt.yticks([])
    # plt.xlim(-0.005, rveBounds[0][1]+.005)
    # plt.ylim(-0.005, rveBounds[1][1]+.005)
    ax.set_aspect('equal')
    plt.savefig(f"../meshes/plots/cur_graph.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300, format = 'pdf')


def edgeAugmentation(nodes, graph_edge, edge_vector, settings, ipCoords, rveBounds):
    """
    Creates edge augmentations, based on: "Gladstone, Rini Jasmine, et al. "GNN-based physics solver for time-independent PDEs." arXiv preprint arXiv:2303.15681 (2023)."
    Not discussed in our resulting paper.

    Adds 'augmentation_percentage' of random augmented edges
    2 Nodes are random uniformly selected. If they don't share an edge yet, an augmented edge is created.
    This is repeated until augmentation_percentage % of edges are augmented.
    These are bi-directional edges, so if Eij is created, so is Eji

    Mode "DistFeats", as presented in original paper above
    Normal edges:       [dx, dy, 0]
    Augmented edges:    [dx, dy, 1]

    Mode "NoFeats", alternative tried here to avoid going out of distribution when scaling to larger meshes
    Normal edges:       [dx, dy, 0]
    Augmented edges:    [0, 0, 1]

    :return:
    """
    # Obtain current number of edges
    start_num_edges = len(graph_edge)

    # Add one-hot feature of 0 to current edge features
    edge_vector = np.hstack((edge_vector, np.zeros((edge_vector.shape[0], 1))))

    # Add additional augmented edges until some % of edges are augmented
    while len(graph_edge) < start_num_edges * (1+settings['augmentation_percentage']):
        # Pick 2 random nodes
        node_a = random.randint(1, nodes)
        node_b = random.randint(1, nodes)
        if node_a == node_b:    # Ignore if nodes are the same
            continue

        # Check if they are not already connected
        if [node_a, node_b] not in graph_edge:
            # Add bi-directional edge
            graph_edge.append([node_a, node_b])
            graph_edge.append([node_b, node_a])

            if settings['edge_augmentation'] == 'DistFeats':
                # Add [dx, dy, 1] for augmented edges. distances for 2 new points to existing feature vector as well as one-hot feature
                # Compute distances
                _, edge_features = euclidDist(ipCoords[graph_edge[-2][0] - 1, :], ipCoords[graph_edge[-2][1] - 1, :], rveBounds, False)

                edge_vector = np.append(edge_vector, [[edge_features[0], edge_features[1], 1],[-edge_features[0], -edge_features[1], 1]], axis=0)

            if settings['edge_augmentation'] == 'NoFeats':
                # Add [0,0,1] edge feature (compared to [dx, dy, 0] for normal edges)
                edge_vector = np.append(edge_vector, [[0, 0, 1], [0, 0, 1]], axis=0)

    return graph_edge, edge_vector


"""
Create the graph
"""

def volumeConsistentGraph(meshFile, settings, fiberpositions, plot_Graph=False):
    """
    Function that computes the rveBounds based on a fixed volume fraction and number of fibers, than calls the createGraph
    """
    if fiberpositions.shape[0] != 2:
        print("ERROR (createGraph): fiberpositions must be a (2, Nfib) array; containinig only the x and y coordinate for every fiber.")
        exit()
    leng_const_vfrac = np.sqrt(fiberpositions.shape[1] * np.pi * settings['r_fiber'] ** 2 / settings['vfrac'])
    rveBounds = [[0, leng_const_vfrac], [0, leng_const_vfrac]]
    return createGraph(meshFile, settings, fiberpositions, rveBounds, plot_Graph)

def createGraph(meshFile, settings, fiberpositions, rveBounds, plot_Graph=False):
    """
    Function that creates a graph based on a micromodel mesh
    :param meshFile: the .msh file
    :param fiberpositions: a (2xN_fib) array of
    :param rveBounds: [[left, right], [lower, upper]] bounds of rve
    :param self_loops: True/False
    :param edge_weight_scale: self-loops are 1; other edges are normalized between [0 - edge_weight_scale]
    :param periodicity: True/False
    :return: graph_edge (list of connected nodes), edge_weight (corresponding weights), fib_dist_feats (features w.r.t. fibers)
    """

    ### Sanity checks
    if fiberpositions.shape[0] != 2:
        print("ERROR (createGraph): fiberpositions must be a (2, Nfib) array; containinig only the x and y coordinate for every fiber.")
        exit()
    if len(rveBounds) != 2 or len(rveBounds[0]) != 2:
        print("ERROR (createGraph): rveBounds must be a 2x2 array")
        exit()

    ### Creating the graph
    # Read mesh file
    mesh_node_coords = []
    mesh_elem_nodes = []

    edge_dict = {}
    graph_edge = []
    elem_id = 1
    loop_ids = [[0, 1], [0, 2], [1, 2]]  # Triangular elements -> 3 edges per element
    with open(meshFile) as data:
        for k, line in enumerate(data):
            array = line.split()
            if len(array) == 4:  # Nodes
                mesh_node_coords.append([float(array[1]), float(array[2])])
            if len(array) == 8:  # Elements
                sorted_node_nums = sorted([int(array[-3]), int(array[-2]), int(array[-1])])
                mesh_elem_nodes.append(sorted_node_nums)
                for l in loop_ids:
                    key = f"{sorted_node_nums[l[0]]} {sorted_node_nums[l[1]]}"
                    if key in edge_dict:  # Edge exists by different graph node -> create graph edge
                        graph_edge.append([edge_dict[key], elem_id])
                        graph_edge.append([elem_id, edge_dict[key]])
                        edge_dict[key] = [edge_dict[key], elem_id]  # Used to differentiate between boundary edges
                    else:
                        edge_dict[key] = elem_id
                elem_id += 1

    meshEdges = []

    if settings['periodicity']:
        for edge_nodes, elements  in edge_dict.items():
            edge_nodes = edge_nodes.split()
            edge_nodes = [int(edge_nodes[0]), int(edge_nodes[1])]
            if isinstance(elements, int):  # It has only 1 connected node -> some boundary
                # print(edge_nodes[0])
                EdgeType = checkMeshEdge(mesh_node_coords[edge_nodes[0] - 1], mesh_node_coords[edge_nodes[1] - 1], rveBounds)
                if EdgeType != "None":
                    meshEdges.append([edge_nodes[0] - 1, edge_nodes[1] - 1, elements, EdgeType])

    nodes = len(mesh_elem_nodes)
    mesh_node_coords = np.array(mesh_node_coords)

    # Create Periodic edges
    if settings['periodicity']:
        BoundEdges = {
            'Left': [],
            'Right': [],
            'Bot': [],
            'Top': []
        }

        for i, meshEdge in enumerate(meshEdges):
            center = [ (mesh_node_coords[meshEdge[0]][0] + mesh_node_coords[meshEdge[1]][0]) / 2,
                       (mesh_node_coords[meshEdge[0]][1] + mesh_node_coords[meshEdge[1]][1]) / 2,
                       meshEdge[2]]
            BoundEdges[ meshEdge[3] ].append(center)

        # Check if number of edges on boundaries are the same
        if len(BoundEdges['Left']) != len(BoundEdges['Right']) or len(BoundEdges['Bot']) != len(BoundEdges['Top']):
            print("ERROR (createGraph): Number of elements on boundaries don't coincide; Periodicity will fail.")
            print(f"left: {len(BoundEdges['Left'])}, Right: {len(BoundEdges['Right'])}, Bot: {len(BoundEdges['Bot'])}, Top: {len(BoundEdges['Top'])}")
            exit()
        # Order the BoundEdges
        for sideEdge in BoundEdges:
            if sideEdge == 'Left' or sideEdge == 'Right':  # Sort on y value
                BoundEdges[sideEdge] = np.array(BoundEdges[sideEdge])
                BoundEdges[sideEdge] = BoundEdges[sideEdge][BoundEdges[sideEdge][:, 1].argsort()]
            if sideEdge == 'Bot' or sideEdge == 'Top':  # Sort on x value
                BoundEdges[sideEdge] = np.array(BoundEdges[sideEdge])
                BoundEdges[sideEdge] = BoundEdges[sideEdge][BoundEdges[sideEdge][:, 0].argsort()]

        PBC = []

        for sideEdge in BoundEdges:
            for i in range(len(BoundEdges[sideEdge])):
                if sideEdge == "Left":
                    graph_edge.append([int(BoundEdges[sideEdge][i, 2]), int(BoundEdges["Right"][i, 2])])
                    PBC.append(len(graph_edge)-1)
                    graph_edge.append([int(BoundEdges["Right"][i, 2]),  int(BoundEdges[sideEdge][i, 2])])
                    PBC.append(len(graph_edge)-1)
                if sideEdge == "Bot":
                    graph_edge.append([int(BoundEdges[sideEdge][i, 2]), int(BoundEdges["Top"][i, 2])])
                    PBC.append(len(graph_edge)-1)
                    graph_edge.append([int(BoundEdges["Top"][i, 2]),    int(BoundEdges[sideEdge][i, 2])])
                    PBC.append(len(graph_edge)-1)
    num_edges = len(graph_edge)

    ipCoords = np.empty((nodes,2))
    ipArea = np.empty( nodes )

    for i in range(nodes):
        ipCoords[i, :], ipArea[i] = elemCenter(mesh_node_coords, mesh_elem_nodes[i])

    # Edge weights as INVERSE Euclidean distance
    if settings['self_loops']:
        edge_weight = np.empty(num_edges+nodes)
    else:
        edge_weight = np.empty(num_edges)

    edge_vector = np.empty_like(graph_edge, dtype=float)

    PBC_index = 0
    for i, edge in enumerate(graph_edge):
        PBC_edge = False
        if settings['periodicity']:
            if i == PBC[PBC_index]:  # Periodic boundary condition
                PBC_edge = True
                PBC_index += 1

        edge_weight[i], edge_vector[i] = euclidDist(ipCoords[edge[0] - 1, :], ipCoords[edge[1] - 1, :], rveBounds, PBC_edge)



    # # Normalize edge_weight (Motivation: might benefit transfer learning)
    # # Edit: normalizing 1 mesh at a time based on its own longest edge creates weird features; where different lengths can be the same normalized value
    # edge_weight = np.array(edge_weight)
    # edge_weight = edge_weight / max(edge_weight)    # NOTE: smallest edge != 0, only scaled such that max edge = 1; to keep linear behaviour

    # Self loops: scale incoming vs self-loop messages. (Self-loops not used in in final paper)
    if settings['self_loops']:
        edge_weight *= settings['edge_weight_scale']
        for i in range(nodes):
            graph_edge.append([int(i+1), int(i+1)])
            edge_weight[num_edges+i] = 1.0

    if settings['edge_augmentation'] != "None":
        # Add edges & edge features
        graph_edge, edge_vector = edgeAugmentation(nodes, graph_edge, edge_vector, settings, ipCoords, rveBounds)

    fib_dist_feats = knn_fibdists(fiberpositions[0], fiberpositions[1], ipCoords, rveBounds, settings['fibers_considered'][0])


    if plot_Graph:
        plotGraph(ipCoords, graph_edge, settings['periodicity'], PBC, rveBounds)


    if settings['edge_vec']:
        return graph_edge, edge_vector, fib_dist_feats, nodes, mesh_elem_nodes, mesh_node_coords, ipArea
    else:
        return graph_edge, edge_weight, fib_dist_feats, nodes, mesh_elem_nodes, mesh_node_coords, ipArea

"""
Test functions
"""

# def testGraph():
#     from readConfig import read_config
#     import polars
#     print("Running testGraph")
#
#     meshdir = '../meshes/fib1-9_mesh6000_t25_4'
#     settings_file = '../fem/ConfigGNN'
#     settings = read_config(settings_file)
#
#     meshnum = 11
#     meshfile = f'{meshdir}/m_{meshnum}.msh'
#
#     # fibers = polars.read_csv(meshdir + '/fiber_coords', separator=' ', has_header=False).to_numpy()
#     # fibers = fibers.tolist()
#
#     fibers = polars.read_csv(meshdir + '/fiber_coords_expanded', separator=',', has_header=False).to_numpy()
#     fibers_set = []
#     for fib_list in fibers:
#         fibers_set.append(fib_list[~np.isnan(fib_list)].tolist())
#
#     curfibers = np.array(fibers_set[meshnum])
#     fiberpositions = curfibers[(np.arange(curfibers.shape[0]) % (3)) < 2].reshape(-1, 2).T  # Extract x and y, without r
#
#     print(fiberpositions)
#
#     graph_edges, edge_feats, fib_feats, num_nodes, mesh_elem_nodes, mesh_node_coords, ipArea = volumeConsistentGraph(meshfile, settings, fiberpositions, plot_Graph=True)
#
#     print(graph_edges)
#     # print(edge_feats)
#     # print(fib_feats)
#
#     return
# testGraph()