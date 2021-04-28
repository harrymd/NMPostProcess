from glob import glob
import os

import numpy as np

from NMPostProcess.common import mkdir_if_not_exist

def pre_process_legacy(dir_PM, dir_NM):
    '''
    Perform some processing steps which only need to be done once for a given NormalModes run.
    See README.md for more information.
    '''

    ellipticity_profile = None

    # Create the output directories if they do not exist.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    for dir_ in [dir_processed, dir_spectral]:
        mkdir_if_not_exist(dir_)

    # Write the eigenvalue file if it doesn't already exist.
    path_eigval_list    = os.path.join(dir_processed, 'eigenvalue_list.txt')
    if not os.path.exists(path_eigval_list):
        
        # Search for the standard output file, which contains the eigenvalue information.
        #file_std_out_wc = os.path.join(dir_NM, '{:}*.txt'.format(std_out_prefix))
        path_eigs_wc = os.path.join(dir_NM, '*.txt')
        path_eigs_glob = glob(path_eigs_wc)
        n_glob = len(path_eigs_glob)
        if n_glob == 0:

            raise FileNotFoundError('Could not find any paths matching {} for eigenvalue information.'.format(path_eigs_wc))

        elif n_glob > 1:
            
            print(path_eigs_glob)
            raise RuntimeError('Found more than one path matching {}, not clear which contains eigenvalue information.'.format(path_eigs_wc))

        path_eigs = path_eigs_glob[0]
    
        # Case 1: The eigenvalue log ends in eigs.txt.
        if path_eigs[-8:-4] == 'eigs':

            path_eig_txt = path_eigs
            write_eigenvalues_from_eigs_txt_file(path_eig_txt, path_eigval_list)

        # Case 2: The eigenvalue log ends in seven integers then .txt (the format of a standard output file). 
        elif all([character in list('0123456789') for character in list(path_eigs[-11:-4])]):

            path_std_out = path_eigs
            write_eigenvalues_from_std_out_file(path_std_out, path_eigval_list = path_eigval_list)
        
        else:

            raise RunTimeError('The name of the raw eigenvalue log file ({:}) does not match any known formats.'.format(path_eigs))

    else:

        print('Eigenvalue list file {:} already exists. Skipping creation.'.format(path_eigval_list))

    # Write the nodes file if it doesn't already exist.
    # This file is simply a re-written version of some of the TetGen output, for convenience.
    path_nodes = os.path.join(dir_processed, 'nodes.txt')
    #if not os.path.exists(path_nodes):
        
        #nodes, tets, tet_means, tet_attrib, neighs = read_mesh(file_base)
        #nodes, _, _, _, _ = read_mesh(dir_PM)
    nodes, node_bd_flag, _, _, _, _, face_list, face_attb, face_tets = read_mesh_legacy(dir_PM)
    print('Saving node file {:}'.format(path_nodes))
    np.savetxt(path_nodes, nodes)

    #else:
    #    
    #    print('Node file already exists: {:}. Skipping creation.'.format(path_nodes))
    #    nodes = np.loadtxt(path_nodes)

    # Write the region index files if they don't already exist.
    path_index_lists = os.path.join(dir_processed, 'index_lists.txt')
    if not os.path.exists(path_index_lists):

        # Read the discontinuity radius information.
        path_discon_info = os.path.join(dir_PM, 'radii.txt')
        r_discons, state_outer = read_discon_file(path_discon_info)

        # Load sample index and attribute information.
        node_idxs, node_attbs, i_first_order = load_sample_indices_and_attribs(dir_NM)
    
        r_nodes = np.linalg.norm(nodes, axis = 1)
        i_outer_node = np.where(np.abs(r_nodes - r_discons[0]) < 1.0E-3)[0]
        i_outer_node = np.sort(i_outer_node)

        # Get list of indices of samples in regions (interiors and boundaries).
        _, _, index_lists = get_indices_of_regions(nodes, node_attbs, node_idxs, r_discons, state_outer, i_outer_node, ellipticity_profile = ellipticity_profile)

        # Save the index list.
        print('Saving index lists file: {:}'.format(path_index_lists))
        index_list_lengths = [len(x) for x in index_lists]
        n_regions = len(index_list_lengths)
        index_list_header_fmt = '{:>12d}'*n_regions + '\n'
        index_list_header = index_list_header_fmt.format(*index_list_lengths)

        with open(path_index_lists, 'w') as out_id:

            out_id.write(index_list_header)

            for index_list in index_lists:
                
                out_id.write('#\n')
                np.savetxt(out_id, index_list, fmt = '%i')

    else:

        print('Detected index list file {:}. Skipping calculation.'.format(path_index_lists))
    
    #for index_list in index_lists:
    #    print(len(index_list))

    return

def read_mesh_legacy(dir_PM):
    '''
    Reads the various TetGen files related to the mesh.
    
    Input

    path_base The path to the TetGen files, including the model prefix, e.g. '/path/to/mod.1'.

    Output

    nodes       See 'Definitions of variables'.
    tets        See TetGen manual.
    tet_means   Centroids of tetrahedra.
    tet_attrib  See TetGen manual.
    neighs      See TetGen manual.
    '''

    # Read the .ele file, which contains a list of tetrahedra. 
    # (See section 5.2.4 of the TetGen manual.)
    # [n_tet]   The number of tetrahedra. 
    # tets      (n_tet, 4) Each row gives the indices of the four vertices of
    #           one of the tetrahedra.
    # tet_attrib(n_tet) A tetrahedron has an 'attribute' (integer code) which
    #           can be used to identify different regions of the model.
    path_ele_regex = os.path.join(dir_PM, '*.ele')
    path_ele_regex_glob = glob(path_ele_regex)
    path_ele_regex_glob = [x for x in path_ele_regex_glob if x[-6:] != '.b.ele']
    assert len(path_ele_regex_glob) == 1, 'Found more than one .ele file (excluding .b.ele).'
    path_ele = path_ele_regex_glob[0]
    mesh_info = np.loadtxt(
                    path_ele,
                    comments    = '#',
                    skiprows    = 1,
                    usecols     = (1, 2, 3, 4, 5),
                    dtype       = np.int,)

    # Note: Here switch to 0-based indexing.
    tets        = mesh_info[:, 0:4] - 1
    tet_attrib  = mesh_info[:, 4]

    # Read the .node file, which contains a list of nodes.
    # (See section 5.2.1 of the TetGen manual.)
    # [n_node]  The number of nodes.
    # nodes     (n_nodes, 3) The coordinates of each node.
    path_node_regex = os.path.join(dir_PM, '*.node')
    path_node_regex_glob = glob(path_node_regex)
    path_node_regex_glob = [x for x in path_node_regex_glob if x[-7:] != '.b.node']
    assert len(path_node_regex_glob) == 1, 'Found more than one .node file (excluding .b.node).'
    path_node = path_node_regex_glob[0]
    node_info       = np.loadtxt(
                    path_node,
                    comments    = '#',
                    skiprows    = 1,
                    usecols     = (1, 2, 3),
                    dtype       = np.float)
    nodes = node_info[:, 0:3]
    node_bd_flag = None
    
    # Find the mean of the corners of each tetrahedron.
    tet_means =     (   nodes[tets[:, 0]]
                    +   nodes[tets[:, 1]]
                    +   nodes[tets[:, 2]]
                    +   nodes[tets[:, 3]])/4.0
                    
    # Read the .neigh file, which lists the four neighbours (which share faces) of each tetrahedron. (Boundary faces have an index of -1.)
    # (See section 5.2.10 of the TetGen manual.)
    # [n_tet]   The number of tetrahedra.
    # neighs    (n_tet, 4) The indices of the neighbours of each tetrahedron.
    path_neigh_regex = os.path.join(dir_PM, '*.neigh')
    path_neigh = glob(path_neigh_regex)[0]
    neighs      = np.loadtxt(
                    path_neigh,
                    comments    = '#',
                    skiprows    = 1,
                    usecols     = (1, 2, 3, 4),
                    dtype       = np.int)
    # Note: Here switch to 0-based indexing.
    neighs = neighs - 1

    # Read the .face file.
    face_list = None
    face_attb = None
    face_tets = None

    return nodes, node_bd_flag, tets, tet_means, tet_attrib, neighs, face_list, face_attb, face_tets
