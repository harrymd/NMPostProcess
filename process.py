'''
NMPostProcess/process.py
Functions for processing NormalModes output with vector spherical harmonics (VSHs).
For usage information, see README.md.

Definitions of variables:


'''

# Import modules --------------------------------------------------------------
# Import core modules.
from functools import partial
from glob import glob
import multiprocessing
import os

# Import third-party modules
import numpy as np
from scipy.interpolate import LinearNDInterpolator 
import shtns
import ssrfpy

# Import local modules.
from common import get_list_of_modes_from_output_files, mkdir_if_not_exist, read_eigenvalues

# Pre-processing steps (only performed once for a given run). -----------------
def write_eigenvalues(file_std_out, file_eigval_list = None): 
    '''
    Reads the eigenvalues (frequencies) from standard output file from the NormalModes code and (optionally) saves them in a separate file. 

    Input

    file_std_out, file_eigval_list (optional)
        See 'Definitions of variables'.

    Returns

    None
    '''

    # Skip lines until the relevant section is reached. 
    target_line = 'Transform to frequencies (mHz), and periods (s)' 
    with open(file_std_out, 'r') as in_id: 
         
        for line in in_id:
            
            line = line.strip()

            if line == target_line:

                break
        
        if not (line == target_line):
            raise ValueError('Reached end of file without finding target line')
        
        line = in_id.readline().split()
        # Create output arrays for mode number and frequency.
        i_mode     = []
        freq    = []
        while len(line) == 4:
            
            i_mode.append(int(line[1]))
            freq.append(float(line[2]))
            
            line = in_id.readline().split()
            
    # Write to eigenvalue file (if requested).
    if file_eigval_list is not None:
        
        print('Writing eigenvalue list file {:}'.format(file_eigval_list))
        with open(file_eigval_list, 'w') as out_id:
            
            for n, f in zip(i_mode, freq):
                
                out_id.write('{:10d} {:16.9f}\n'.format(n, f))
                
    return i_mode, freq

def read_discon_file(path_discon_info):
    '''
    Reads information about solid-fluid discontinuities.

    Input:

    path_discon_info
        See 'Definitions of variables'.

    Output:

    r_discons, state_outer
        See 'Definitions of variables'.
    '''

    with open(path_discon_info, 'r') as in_id:

        # Read lines, remove newline character.
        lines = in_id.readlines()
        lines = [x.strip() for x in lines]

        # First line is state of outer shell.
        state_outer = lines[0]

        # Remaining lines are radii of discontinuities (including surface), starting at the surface.
        r_discons = np.array([float(x) for x in lines[1:]])

    return r_discons, state_outer
    
def get_indices_of_regions(nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default'):
    '''
    For each region (outer surface, interior, and inner surface of each shell), find the indices of the samples belonging to that region.

    Input:

    See 'Definitions of variables.'

    Output:

    See 'Definitions of variables.'
    '''
    
    # Get the radius of the surface and the number of discontinuities.
    r_surface = r_discons[0]
    n_discons = len(r_discons)

    # Get the radial coordinates of the samples.
    r_nodes = np.linalg.norm(nodes, axis = 1)
    r_samples = r_nodes[node_idxs]

    # Find nodes on the free surface.
    i_shell = 0
    is_outer = True
    surface_condition, j_surface = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer)

    # Find all interior nodes (must remove the surface nodes).
    interior_condition = (((node_attbs == 0) | (node_attbs == 1)) & ~surface_condition)
    j_interior_sample = np.where(interior_condition)[0]

    # Case 1: There are no interior discontinuities.
    if n_discons == 1:
        
        # Find samples inside the object.
        index_lists_interior = [j_interior_sample]
        
        # Find samples on the surface of the object.
        index_lists_boundaries = [j_surface]

    # Case 2: There are some interior discontinuities.
    else:

        # Initialise lists.
        index_lists_interior = []
        index_lists_boundaries = [j_surface]

        # Make a list of boundaries of shells (r_surface, r1, r2, ..., 0.0).
        r_discons_bounded = np.concatenate([r_discons, [0.0]])

        # Find the indices of interior samples in each shell.
        for i_shell in range(n_discons):
            
            # Find samples at the appropriate radial distance.
            radius_condition = ((r_samples < r_discons_bounded[i_shell]) & (r_samples > r_discons_bounded[i_shell + 1]))

            # Apply the radial condition and the interior node attribute condition.
            shell_condition = (radius_condition & interior_condition)
            j_shell = np.where(shell_condition)[0]
            
            # Store.
            index_lists_interior.append(j_shell)

        # Find the indices of boundary samples in each shell.
        for i_shell in range(n_discons):

            # Loop over the outer and inner sides of each shell.
            # The outer surface of the outer shell has already been found.
            if i_shell == 0:

                is_outer_list = [False]

            elif (i_shell < (n_discons - 1)):

                is_outer_list = [True, False]

            # The innermost shell does not have an inner side.
            else:

                is_outer_list = [True]

            for is_outer in is_outer_list:
                
                # Search for the boundary samples.
                _, j_discon = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')

                index_lists_boundaries.append(j_discon)

    # Merge the lists into a single list.
    # The list starts with the free surface and then the first interior region.
    index_lists = [index_lists_boundaries[0], index_lists_interior[0]]
    if len(index_lists_interior) > 1:

        for i in range(1, len(index_lists_interior)):

            # Between every interior region, there are two boundaries.
            index_lists.append(index_lists_boundaries[2*i - 1])
            index_lists.append(index_lists_boundaries[2*i])

            # The next interior region.
            index_lists.append(index_lists_interior[i])

    return index_lists_interior, index_lists_boundaries, index_lists

def get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default'):
    '''
    Finds the samples of the eigvector which belong to a specified boundary.
    
    Input:

    See 'Definitions of variables.'

    Output:

    condition
        (n_samples) True for the samples on the boundary, False otherwise. Therefore j_boundary = np.where(condition)[0].
    j_boundary
        The indices of the samples which belong to the boundary.
    '''

    # Case 1: The interface is the outer surface.
    if (i_shell == 0) and is_outer:

        # Find radius of outer surface.
        r_surface = r_discons[0]

        # Set default value of comparison tolerance for boundaries.
        if boundary_tol == 'default':

            boundary_tol = r_surface*1.0E-7

        # Find indices of allowed nodes, and discard other nodes
        r_samples = np.linalg.norm(nodes[node_idxs], axis = 1)
        r_min = r_surface - boundary_tol
        condition = (r_samples > r_min)

    # Case 2: The interface is an internal fluid-solid boundary.
    else:
        
        # Find the index of the region the sample belongs to.
        if is_outer:
            
            i_region_discon = i_shell

        else:

            i_region_discon = i_shell + 1

        # Get discon lists.
        r_discon_midpoints, state_list, n_interior_discons = get_discon_info(r_discons, state_outer)
        
        if n_interior_discons > 1:

            # Find which nodes are close to the specified discontinuity. 
            r_nodes = np.linalg.norm(nodes, axis = 1)
            r_samples = r_nodes[node_idxs]
            radius_condition = ((r_samples < r_discon_midpoints[i_region_discon - 1]) & (r_samples > r_discon_midpoints[i_region_discon]))

        else:
            
            # If there is only one internal discontinuity, we do not have to specify a radius condition to identify it.
            radius_condition = True

        # Case 2a. The query point is in a solid shell.
        if state_list[i_shell] == 'solid':

            attb_condition = (node_attbs == 2)

        # Case 2b. The query point is in a liquid shell.
        else:

            attb_condition = (node_attbs == 6)

        # Apply the attribute and radius conditions simultaneously.
        condition = (attb_condition & radius_condition)
        
    j_boundary = np.where(condition)[0]

    return condition, j_boundary

def pre_process(dir_PM, dir_NM):
    '''
    Perform some processing steps which only need to be done once for a given NormalModes run.
    See README.md for more information.
    '''

    # Create the output directories if they do not exist.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    for dir_ in [dir_processed, dir_spectral]:
        mkdir_if_not_exist(dir_)

    # Write the eigenvalue file if it doesn't already exist.
    file_eigval_list    = os.path.join(dir_processed, 'eigenvalue_list.txt')
    if not os.path.exists(file_eigval_list):
        
        # Search for the standard output file, which contains the eigenvalue information.
        #file_std_out_wc = os.path.join(dir_NM, '{:}*.txt'.format(std_out_prefix))
        file_std_out_wc = os.path.join(dir_NM, '*.txt')
        file_std_out_glob = glob(file_std_out_wc)
        n_glob = len(file_std_out_glob)
        if n_glob == 0:

            raise FileNotFoundError('Could not find any files matching {}'.format(file_std_out_wc))

        elif n_glob > 1:
            
            raise RuntimeError('Found more than one file matching {}'.format(file_std_out_wc))

        file_std_out = file_std_out_glob[0]

        # Read the standard output file and write the eigenvalue file.
        write_eigenvalues(file_std_out, file_eigval_list = file_eigval_list)

    else:

        print('Eigenvalue list file {:} already exists. Skipping creation.'.format(file_eigval_list))

    # Write the nodes file if it doesn't already exist.
    # This file is simply a re-written version of some of the TetGen output, for convenience.
    path_nodes = os.path.join(dir_processed, 'nodes.txt')
    if not os.path.exists(path_nodes):
        
        #nodes, tets, tet_means, tet_attrib, neighs = read_mesh(file_base)
        nodes, _, _, _, _ = read_mesh(dir_PM)
        print('Saving node file {:}'.format(path_nodes))
        np.savetxt(path_nodes, nodes)

    else:
        
        print('Node file already exists: {:}. Skipping creation.'.format(path_nodes))
        nodes = np.loadtxt(path_nodes)

    # Write the region index files if they don't already exist.
    path_index_lists = os.path.join(dir_processed, 'index_lists.txt')
    if not os.path.exists(path_index_lists):

        # Read the discontinuity radius information.
        path_discon_info = os.path.join(dir_PM, 'radii.txt')
        r_discons, state_outer = read_discon_file(path_discon_info)

        # Load sample index and attribute information.
        node_idxs, node_attbs = load_sample_indices_and_attribs(dir_NM)
            
        # Get list of indices of samples in regions (interiors and boundaries).
        _, _, index_lists = get_indices_of_regions(nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')

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

    return

# Reading NormalModes files. --------------------------------------------------
def read_mesh(dir_PM):
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
    path_ele = glob(path_ele_regex)[0]
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
    path_node = glob(path_node_regex)[0]
    nodes       = np.loadtxt(
                    path_node,
                    comments    = '#',
                    skiprows    = 1,
                    usecols     = (1, 2, 3),
                    dtype       = np.float)
                    
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

    return nodes, tets, tet_means, tet_attrib, neighs

def load_sample_indices_and_attribs(dir_NM):
    '''
    Read the *_vlist.dat and *_vstat.dat files, and slightly reformat them to help to distinguish between samples on the inner and outer surfaces of discontinuities.

    Input:

    dir_NM
        See 'Definitions of variables'.

    Output:

    node_idxs, node_attbs
        See 'Definitions of variables'.
    '''
    
    # Read the vlist file (a list of vertex indices).
    path_vlist_regex = os.path.join(dir_NM, '*_vlist.dat')
    path_vlist = glob(path_vlist_regex)[0]
    node_idxs   = np.fromfile(path_vlist, dtype = '<i')
    # Convert to 0-based indexing.
    node_idxs   = node_idxs - 1
    
    # Read the vstat file (a list of vertex attributes).
    path_vstat_regex = os.path.join(dir_NM, '*_vstat.dat')
    path_vstat = glob(path_vstat_regex)[0]
    node_attbs  = np.fromfile(path_vstat, dtype = '<i')
    
    # Print a vertex attribute summary.
    n_nodes_by_attb = [np.sum(node_attbs == i) for i in range(3)]
    print('Vertex attribute summary:')
    for i in range(3):

        print('{:2d} {:6d}'.format(i, n_nodes_by_attb[i]))
    
    # Calculate the number of nodes and the number of samples.
    # (See 'Definitions of variables' for an explanation.)
    n_nodes = node_idxs.shape[0]
    n_samples = n_nodes_by_attb[0] + n_nodes_by_attb[1] + 2*n_nodes_by_attb[2]

    # Insert a new vertex attribute for the each of the fluid-solid boundary
    # nodes, so we can distinguish between the eigenvector on the fluid side
    # and the solid side.
    node_attbs_new  = np.zeros(n_samples, dtype = node_attbs.dtype)
    node_idxs_new   = np.zeros(n_samples, dtype = node_idxs.dtype)
    j = 0
    for i in range(n_nodes):

        # For interior and free-surface nodes, the indexing does not need to be
        # changed.
        node_attbs_new[j]   = node_attbs[i]
        node_idxs_new[j]    = node_idxs[i]
        
        # A vertex attribute of 2 or 5 indicates a fluid-solid boundary node.
        # (2 is first-order, 5 is second-order.)
        if (node_attbs[i] == 2) or (node_attbs[i] == 5):

            j = j + 1

            # Repeat the node index (displacement is specified twice at this
            # node).
            node_idxs_new[j]    = node_idxs[i]
            
            # For the repeated node, set a new attribute to indicate it is on
            # the liquid side of the boundary.
            if node_attbs[i] == 2:

                node_attbs_new[j] = 6

            elif node_attbs[i] == 5:

                node_attbs_new[j] = 7

        j = j + 1
        
    node_idxs   = node_idxs_new
    node_attbs  = node_attbs_new

    return node_idxs, node_attbs

def read_eigenvector(i_mode, path_eigvec_base):
    '''
    Reads a single eigenfunction from a .dat file output by NormalModes.
    
    Input:
    
    See 'Definitions of variables'.

    Output:

    See 'Definitions of variables'.
    '''
    
    # Read an eigenvector (a flattened list of vector displacements).
    path_eigvec = '{:}_{:d}.dat'.format(path_eigvec_base, i_mode)
    eigvec_flat= np.fromfile(path_eigvec, dtype = 'float64')
    
    # Put the eigenvector in the form (n, 3).
    n_eigvec_flat = len(eigvec_flat)
    n_eigvec_flat_mod_3 = (n_eigvec_flat % 3)
    if n_eigvec_flat_mod_3 != 0:

        raise ValueError('Size of flattened eigenvector array is not a multiple of 3.')
    #    
    eigvec     = eigvec_flat.reshape((int(len(eigvec_flat)/3), 3))
  
    return eigvec

def read_mode(dir_NM, i_mode, path_eigvec_base):
    '''
    Read information about one mode.

    Input:

    dir_PM, dir_NM, i_mode
        See 'Definitions of variables'.

    Output:

    freq, nodes, node_idxs, node_attbs, eigvec
        See 'Definitions of variables'.
    '''
    
    # Read the eigenvalue of this mode.
    dir_processed           = os.path.join(dir_NM, 'processed')
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    i_mode_list, freq_list  = read_eigenvalues(file_eigval_list)
    i_mode_list             = np.array(i_mode_list, dtype = np.int)
    freq_list               = np.array(freq_list)
    freq                    = freq_list[np.where(i_mode_list == i_mode)[0][0]]
   
    # Read the eigenvector.
    eigvec = read_eigenvector(i_mode, path_eigvec_base) 

    return freq, eigvec

# Reading NMPostProcess files. ------------------------------------------------
def read_index_lists(dir_NM):
    '''
    Read the index lists file generated by get_indices_of_regions().

    Input:

    dir_NM
        See 'Definitions of variables'.

    Output:

    index_lists
        See 'Definitions of variables'.
    '''

    # Find path to index lists file.
    dir_processed = os.path.join(dir_NM, 'processed')
    path_index_lists = os.path.join(dir_processed, 'index_lists.txt')

    # Read the header.
    with open(path_index_lists, 'r') as in_id:

        header = in_id.readline()

    # Get the index list lengths from the header.
    index_list_lengths = [int(x) for x in header.split()]

    # Read the index lists.
    index_lists_raw = np.loadtxt(path_index_lists, dtype = np.int, skiprows = 1, comments = '#')

    # Separate the index lists.
    index_lists = []
    index_list_length_cumulative = 0
    for index_list_length in index_list_lengths:
        
        i0 = index_list_length_cumulative
        i1 = i0 + index_list_length
        index_lists.append(index_lists_raw[i0 : i1])

        index_list_length_cumulative = index_list_length_cumulative + index_list_length
    
    return index_lists

def read_info_for_projection(dir_PM, dir_NM):
    '''
    Read information about the mesh required for doing projection.

    Input:

    dir_PM, dir_NM
        See 'Definitions of variables'.

    Returns:

    nodes, node_idxs, node_attbs, r_discons, state_outer, n_discons, index_lists
        See 'Definitions of variables'.
    
    '''

    # Read node coordinates.
    dir_processed = os.path.join(dir_NM, 'processed')
    path_nodes = os.path.join(dir_processed, 'nodes.txt')
    nodes = np.loadtxt(path_nodes)

    # Read sample indices and attributes.
    node_idxs, node_attbs = load_sample_indices_and_attribs(dir_NM)

    # Read the discontinuity radius information.
    path_discon_info = os.path.join(dir_PM, 'radii.txt')
    r_discons, state_outer = read_discon_file(path_discon_info)
    n_discons = len(r_discons)

    # Read lists of indices of samples in interior regions and on discontinuities.
    index_lists = read_index_lists(dir_NM) 

    return nodes, node_idxs, node_attbs, r_discons, state_outer, n_discons, index_lists 

def get_eigvec_path_base(dir_NM):
    '''
    Determine the common root path for all of the NM output files.

    Input:

    dir_NM
        See 'Definitions of variables'.


    Output:

    eigvec_path_base
        See 'Defintions of variables'.
    '''

    # Look for all files in the output directory ending in *.dat.
    regex_eigvec = '*.dat'
    path_regex_eigvec = os.path.join(dir_NM, regex_eigvec)
    eigvec_path_list = glob(path_regex_eigvec)
    #
    for eigvec_path in eigvec_path_list:

        # Remove .dat suffix.
        eigvec_path = eigvec_path[:-4]

        # Check this is a mode file (ends in an integer).
        eigvec_path_split = eigvec_path.split('_')
        try:

            i_mode = int(eigvec_path_split[-1])
            eigvec_path_base = '_'.join(eigvec_path_split[:-1])

            # Return the first matching pattern.
            return eigvec_path_base

        except ValueError:

            pass

    raise ValueError('No files of correct file name format found in directory {:}'.format(dir_NM))

# Generic functions. ----------------------------------------------------------
def xyz_to_rlonlat(x, y, z):
    '''
    Converts from Cartesian coordinates to radius, longitude and latitude.
    https://mathworld.wolfram.com/SphericalCoordinates.html

    Input:

    x, y, z
        The Cartesian coordinates.

    Returns:

    r   Radial coordinate.
    lon Longitude (radians between -pi and pi).
    lat Latitude (radians between -pi/2 and pi/2).
    '''

    r       = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta   = np.arccos(z/r)
    lat     = (np.pi/2.0) - theta
    lon     = np.arctan2(y, x)

    return r, lon, lat

def rlonlat_to_xyz(r, lon, lat):
    '''
    Converts from radius, longitude and latitude to Cartesian coordinates.
    https://mathworld.wolfram.com/SphericalCoordinates.html

    Input:

    r   Radial coordinate.
    lon Longitude (radians).
    lat Latitude (radians).

    Returns:

    x, y, z
        The Cartesian coordinates.
    '''

    # Theta: Angle downward from z axis.
    theta = (np.pi/2.0) - lat
        
    x = r*np.sin(theta)*np.cos(lon)
    y = r*np.sin(theta)*np.sin(lon)
    z = r*np.cos(theta)
    
    return x, y, z

# Common tools for NMPostProcess. --------------------------------------------
def get_discon_info(r_discons, state_outer):
    '''
    Constructs useful arrays based on discontinuity information.

    Input:

    r_discons, r_surface
        See 'Definitions of variables'.

    Returns:
    
    r_discon_midpoints, state_list, n_interior_discons
        See 'Definitions of variables'.
    '''
    
    r_surface = r_discons[0]
    r_interior_discons = r_discons[1:]

    # Count the number of shells (for example, Earth has three shells: solid inner core, fluid outer core, and solid mantle).
    n_interior_discons = len(r_interior_discons)
    
    # Make a list of the mid-points between discontinuities.
    if n_interior_discons > 1:
        
        r_discon_midpoints = 0.5*(r_interior_discons[:-1] + r_interior_discons[1:])

        # Add end points for bracketing.
        r_discon_midpoints = np.concatenate([[r_surface], r_discon_midpoints, [0.0]])

    else:

        r_discon_midpoints = None

    # Create a list of the state of each shell (for example, Earth's is ['solid', 'liquid', 'solid'].
    if state_outer == 'solid':

        i_offset = 0

    else:

        i_offset = 1
    
    state_list = []
    n_shells = n_interior_discons + 1
    for i in range(n_shells):

        if ((i + i_offset) % 2 == 0):

            state_list.append('solid')

        else:

            state_list.append('liquid')

    return r_discon_midpoints, state_list, n_interior_discons

def find_r_max(nodes, node_idxs, eigvec, index_lists, r_min = None):
    '''
    For a specified displacement field, find the radius at which the maximum displacement occurs.

    Input

    nodes, node_idxs, eigvec, index_lists 
        See 'Definitions of variables'.
    r_min   A minimum radius. Points inside this radius will be ignored. If 'default', the minimum radius will be 10% of the outer radius.

    Output

    r_max, i_region_max
        See 'Definitions of variables.'
    S_max
        The maximum absolute value of displacement.
    '''

    # If a minimum radius is specified, discard points inside this radius.
    if r_min is not None:

        # Find indices of allowed samples, and set other samples to 0.
        r_samples = np.linalg.norm(nodes[node_idxs], axis = 1)
        r_condition = (r_samples < r_min)
        j_ignore = np.where(r_condition)[0]
        eigvec[j_ignore, :] = 0.0
        
    # Calculate amplitude of eigenfunction at each sample. 
    S = np.linalg.norm(eigvec, axis = 1)

    # Find index of sample with greatest amplitude.
    j       = np.argmax(S)
    S_max   = S[j]
    i       = node_idxs[j]
    node_i  = nodes[i, :]
    r_max     = np.linalg.norm(node_i)
    
    # Find which region the maximum-amplitude sample belongs to.
    found = False
    for i_region, index_list in enumerate(index_lists):

        if j in index_list:
            
            found = True
            i_region_max = i_region
            break

    return r_max, i_region_max, S_max

def interpolate_eigvec_onto_sphere(r_q, i_region_q, nodes, node_idxs, eigvec, index_lists, n_lat_grid):
    '''
    Interpolates the 3D displacement field onto a regular grid on a sphere with a specified radius.  

    Input:

    r_q, i_region_q, nodes, node_idxs, eigvec, eigvec, index_lists, n_lat_grid
        See 'Definitions of variables.'
    Note: If i_region_q specifies a discontinuity (case 1), then r_q is ignored.

    Output:

    lon_grid, lat_grid, eigvec_grid
        See 'Definitions of variables.'
    '''

    # Make lists of the regions which are discontinuities and which are interiors.
    n_regions = len(index_lists)
    n_shells = (n_regions + 1)//3
    #
    #i_region_interior = [3*i + 1 for i in range(n_shells)]
    i_region_discon = []
    for i in range(n_shells):

        i_region_discon.append(3*i - 1)
        i_region_discon.append(3*i)

    i_region_discon = i_region_discon[1:]

    # Case 1: The query point is within one of the discontinuities.
    if i_region_q in i_region_discon:

        # Find the points from the given discontinuity.
        j = index_lists[i_region_q]
        eigvec = eigvec[j, :]
        nodes = nodes[node_idxs[j], :]

        # Get the longitude and latitude of the nodes.
        r_nodes, lon_nodes, lat_nodes = xyz_to_rlonlat(*nodes.T) 

        # Use the ssrfpy angle convention (longitude between 0 and 2*pi).
        lon_nodes[lon_nodes < 0.0] = lon_nodes[lon_nodes < 0.0] + 2.0*np.pi

        # Prepare ssrfpy input parameters.
        n_lon_grid      = (2*n_lat_grid) - 1
        ssrfpy_kwargs   = { 'n'             :   n_lat_grid - 1,
                            'method'        :   'linear',
                            'degrees'       :   False,
                            'use_legendre'  :   False}

        # Interpolate onto a regular grid using ssrfpy.
        # Each component is interpolated separately.
        lon_grid, lat_grid, eigvec_x_grid = ssrfpy.interpolate_regular_grid(lon_nodes, lat_nodes, eigvec[:, 0], **ssrfpy_kwargs)
        _, _, eigvec_y_grid = ssrfpy.interpolate_regular_grid(lon_nodes, lat_nodes, eigvec[:, 1], **ssrfpy_kwargs)
        _, _, eigvec_z_grid = ssrfpy.interpolate_regular_grid(lon_nodes, lat_nodes, eigvec[:, 2], **ssrfpy_kwargs)
           
        # Remove extra row added by ssrfpy.
        lon_grid        = lon_grid[:, :-1]
        lat_grid        = lat_grid[:, :-1]
        eigvec_x_grid   = eigvec_x_grid[:, :-1]
        eigvec_y_grid   = eigvec_y_grid[:, :-1] 
        eigvec_z_grid   = eigvec_z_grid[:, :-1]

        # Recombine output into a single array.
        eigvec_grid = np.array([eigvec_x_grid, eigvec_y_grid, eigvec_z_grid])
        eigvec_grid = np.moveaxis(eigvec_grid, 0, -1)

    # Case 2: The query point is within the interior of one of the shells.
    else:
        
        # Extract the nodes from the interior region and its surrounding discontinuities.
        # The inner region only has one bounding discontinuity.
        if i_region_q == (n_regions - 1):
            
            i_region_list = [i_region_q - 1, i_region_q]

        # Other regions have two bounding discontinuities.
        else:

            i_region_list = [i_region_q - 1, i_region_q, i_region_q + 1]
        #
        j = np.concatenate([index_lists[i] for i in i_region_list])
        eigvec = eigvec[j, :]
        nodes = nodes[node_idxs[j], :]

        # Build a regular grid in longitude and latitude.
        # For consistency with Case 1, we use the grid from ssrfpy.interpolate_regular_grid().
        n_lon_grid = 2*n_lat_grid - 1
        lon_span = np.linspace(0.0, 2.0*np.pi, n_lon_grid, endpoint = True)[:-1]
        lat_span = np.linspace(-np.pi/2.0, np.pi/2.0, n_lat_grid, endpoint=True)
        #
        lon_grid, lat_grid = np.meshgrid(lon_span, lat_span)

        # Find the Cartesian coordinates of the lon/lat grid.
        x_grid, y_grid, z_grid = rlonlat_to_xyz(r_q, lon_grid, lat_grid)
        
        # Create a linear interpolation function.
        #if interpolator is None:
        interpolator = LinearNDInterpolator(nodes, eigvec)
           
        # Find the eigenfunction at the grid points using interpolation.
        eigvec_grid = interpolator((x_grid, y_grid, z_grid))

    return lon_grid, lat_grid, eigvec_grid

# Tools for projecting into vector spherical harmonics. -----------------------
def unit_vectors_at_points(lon, lat):
    '''
    Calculates unit coordinate vectors at specified coordinates.

    Input

    [n]     Number of query points
    lon     (n) Longitude of query points.
    lat     (n) Latitude of query points.

    Output

    r_hat   (n, 3) Outward radial unit vector at each of the query points.
    e_hat   (n, 3) Eastward unit vector at each of the query points.
    n_hat   (n, 3) Northward unit vector at each of the query points.
    '''

    # Find shape of output arrays.
    assert lon.shape == lat.shape
    grid_shape = lon.shape
    vector_shape = (*grid_shape, 3)

    # Calculate unit radial vectors.
    r_hat = np.zeros(vector_shape)
    r_hat[..., 0] = np.cos(lon)*np.cos(lat)
    r_hat[..., 1] = np.sin(lon)*np.cos(lat)
    r_hat[..., 2] = np.sin(lat)
    
    # Calculate unit vectors in the eastward direction.
    e_hat       = np.zeros(vector_shape)
    e_hat[..., 0] = -np.sin(lon)
    e_hat[..., 1] =  np.cos(lon)
    
    # Calculate vectors in the northward direction.
    n_hat       = np.zeros(vector_shape)
    n_hat[..., 0] = -np.cos(lon)*np.sin(lat)
    n_hat[..., 1] = -np.sin(lon)*np.sin(lat)
    n_hat[..., 2] =  np.cos(lat)
    
    return r_hat, e_hat, n_hat

def project_along_unit_vectors(s, r_hat, e_hat, n_hat):
    '''
    Find the radial, eastward and northward components of a vector field by taking the dot product of each vector sample with the radial, eastward and northward unit vectors at the sample location.

    Input

    [n]     The number of nodes.
    s       (n, 3) The vector at each node.
    r_hat   (n, 3) The outward radial unit vector at each node.
    e_hat   (n, 3) The eastward unit vector at each node.
    n_hat   (n, 3) The northward unit vector at each node.

    Output

    s_r, s_e, s_n
        (n) The radial, east and north componets of the vector field at each sample point.
    '''

    # Calculate the dot product at each sample point.
    s_r = np.sum(s*r_hat, axis = -1)
    s_e = np.sum(s*e_hat, axis = -1)
    s_n = np.sum(s*n_hat, axis = -1)
        
    return s_r, s_e, s_n

def transform_to_spherical_harmonics(s_r_grid, s_n_grid, s_e_grid, n_lat_grid, n_lon_grid, l_max):
    '''
    Transforms a vector field into vector spherical harmonics.

    Input:

    s_r_grid, s_n_grid, s_e_grid, n_lat_grid, n_lon_grid, l_max
        See 'Definitions of variables'.

    Output:

    Ulm, Vlm, Wlm, l_list, m_list, sh_calculator
        See 'Definitions of variables'.
    '''
    
    # Use SHTns to calculate the vector spherical harmonics.
    #
    # Create an instance of the spherical harmonic transform object and set the grid.
    m_max = l_max
    sh_calculator   = shtns.sht(l_max, m_max)
    grid_type       =       shtns.sht_reg_fast          \
                        |   shtns.SHT_SOUTH_POLE_FIRST  \
                        |   shtns.SHT_PHI_CONTIGUOUS
    sh_calculator.set_grid(n_lat_grid, n_lon_grid, flags = grid_type)
    #
    # Transform the spatial vector components (radial, theta and phi) into vector spherical harmonics.
    #
    # Initialise the output arrays which will contain the coefficients.
    Ulm = sh_calculator.spec_array()
    Vlm = sh_calculator.spec_array()
    Wlm = sh_calculator.spec_array()
    #
    # Calculate the coefficients.
    sh_calculator.spat_to_SHqst(
                np.ascontiguousarray(s_r_grid),
                np.ascontiguousarray(-1.0*s_n_grid),
                np.ascontiguousarray(s_e_grid),
                Ulm, Vlm, Wlm)
    
    # Retrieve the l- and m-values (angular order and degree) corresponding to each coefficient.
    l_list = sh_calculator.l
    m_list = sh_calculator.m
    
    return Ulm, Vlm, Wlm, l_list, m_list, sh_calculator

def project_from_spherical_harmonics(sh_calculator, Ulm, Vlm, Wlm):
    '''
    Calculates the radial, consoidal and toroidal components of a vector field from the vector spherical harmonic coefficients. 

    Input:

    sh_calculator, Ulm, Vlm, Wlm
        See 'Definitions of variables'.

    Output

    Ur, Ve, Vn, We, Wn
        See 'Definitions of variables'.
    '''

    # Do the inverse spherical harmonic transform (i.e. synthesis) of the radial, consoidal and toroidal coefficients separately to get the spatial representation of these parts of the field.
    # Note that zero_spec is used to set the other two components to zero in the calculation, and unwanted output components are sent to zero_spat.
    zero_spec   = sh_calculator.spec_array()
    zero_spat   = sh_calculator.spat_array()
     
    # Get radial part of field.
    Pr          = sh_calculator.spat_array()
    sh_calculator.SHqst_to_spat(
                Ulm, zero_spec, zero_spec,
                Pr, zero_spat, zero_spat)
                
    # Get consoidal part of field.
    Vn = sh_calculator.spat_array()
    Ve = sh_calculator.spat_array()
    sh_calculator.SHqst_to_spat(
                zero_spec, Vlm, zero_spec,
                zero_spat, Vn, Ve)
    # Convert from theta component to north component.
    Vn = Vn*-1.0
    ## Calculate the magnitude of the consoidal field.
    #B   = np.sqrt(Ve**2.0 + Vn**2.0) 
    
    # Get toroidal part of field.
    Wn = sh_calculator.spat_array()
    We = sh_calculator.spat_array()
    sh_calculator.SHqst_to_spat(
                zero_spec, zero_spec, Wlm,
                zero_spat, Wn, We)
    # Convert from theta component to north component.
    Wn = Wn*-1.0
    ## Calculate the magnitude of the toroidal field.
    #C   = np.sqrt(We**2.0 + Wn**2.0)
    
    return Pr, Ve, Vn, We, Wn

# Vector-spherical-harmonic projection at one radius ('quick mode'). ----------
def vsh_projection_quick(dir_PM, dir_NM, l_max, eigvec_path_base, nodes, node_idxs, node_attbs, index_lists, r_discons, i_mode, save_spatial = False):
    '''
    Calculate vector-spherical-harmonic coefficients of displacement field at the radius of maximum displacement.

    Input:

    dir_PM, dir_NM, l_max, eigvec_path_base, nodes_node_idxs, node_attbs, index_lists, r_discons, i_mode, save_spatial
        See 'Definitions of variables'.

    Output:

    None (results are saved to file)
    '''

    # Note: Must be an even number.
    n_lat_grid = 6*l_max

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Read the information about the mode.
    freq, eigvec = read_mode(dir_NM, i_mode, eigvec_path_base)

    # Discard information about second-order nodes.
    i_allowed   = np.where((node_attbs == 0) | (node_attbs == 1) | (node_attbs == 2) | (node_attbs == 6))[0]
    node_idxs   = node_idxs[i_allowed]
    node_attbs  = node_attbs[i_allowed]
    eigvec      = eigvec[i_allowed, :]
    index_lists = [np.intersect1d(x, i_allowed) for x in index_lists]

    # Find the radial coordinate where the largest displacement occurs.
    r_max, i_region_max, eigvec_max = find_r_max(nodes, node_idxs, eigvec, index_lists, r_min = 0.1*r_discons[0])

    # Normalise the eigenvector, for better numerical behavior.
    eigvec = eigvec/eigvec_max

    # At this radial coordinate, interpolate the displacement field onto the sphere.
    lon_grid, lat_grid, eigvec_grid = interpolate_eigvec_onto_sphere(r_max, i_region_max, nodes, node_idxs, eigvec, index_lists, n_lat_grid)
    n_lon_grid = lon_grid.shape[1]

    # At the grid points, calculate unit vectors in the radial, east and north directions (in terms of the Cartesian basis).
    r_hat_grid, e_hat_grid, n_hat_grid = unit_vectors_at_points(lon_grid, lat_grid)

    # Project the vector field into the radial, east and north directions.
    eigvec_r_grid, eigvec_e_grid, eigvec_n_grid = project_along_unit_vectors(eigvec_grid, r_hat_grid, e_hat_grid, n_hat_grid)

    # Transform vector spatial components to vector spherical harmonics.
    Ulm, Vlm, Wlm, l_list, m_list, sh_calculator = \
        transform_to_spherical_harmonics(
            eigvec_r_grid, eigvec_n_grid, eigvec_e_grid, n_lat_grid, n_lon_grid, l_max)

    if save_spatial:

        # Calculate the spatial representation of the radial, consoidal and toroidal components.
        Ur_grid, Ve_grid, Vn_grid, We_grid, Wn_grid = \
            project_from_spherical_harmonics(sh_calculator, Ulm, Vlm, Wlm)

        # Save spatial output.
        array_out_spatial = np.array([eigvec_grid[..., 0], eigvec_grid[..., 1], eigvec_grid[..., 2], eigvec_r_grid, eigvec_e_grid, eigvec_n_grid, Ur_grid, Ve_grid, Vn_grid, We_grid, Wn_grid])
        dir_spatial = os.path.join(dir_processed, 'spatial')
        mkdir_if_not_exist(dir_spatial)
        file_out_spatial = 'quick_spatial_{:>05d}.npy'.format(i_mode)
        path_out_spatial = os.path.join(dir_spatial, file_out_spatial)
        print('Saving spatial data to {:}'.format(path_out_spatial))
        np.save(path_out_spatial, array_out_spatial)

    # Save spectral output.
    array_out_spectral = np.array([Ulm, Vlm, Wlm])
    #
    # Add a header line with the scale information.
    array_out_spectral = np.insert(array_out_spectral, 0, [eigvec_max, 0.0, 0.0], axis = 1)
    #
    # Add header with maximum radius information.
    array_out_spectral = np.insert(array_out_spectral, 0, [r_max, i_region_max, 0.0], axis = 1)
    #
    dir_spectral = os.path.join(dir_processed, 'spectral')
    file_out_spectral = 'quick_spectral_{:>05d}.npy'.format(i_mode)
    path_out_spectral = os.path.join(dir_spectral, file_out_spectral)
    print('Saving spectral data to {:}'.format(path_out_spectral))
    np.save(path_out_spectral, array_out_spectral)

    return

def vsh_projection_quick_wrapper(dir_PM, dir_NM, l_max, i_mode, eigvec_path_base, save_spatial = False):
    '''
    A wrapper for vsh_projection_quick(), which assembles the relevant information.

    Input:

    dir_PM, dir_NM, l_max, i_mode, eigvec_path_base, save_spatial
        See 'Definitions of variables'.

    Output:

    None (output is saved to file).
    '''
    
    #i_mode = np.random.randint(low = 1, high = 83) 

    # Define directory names.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Read various information about the mesh.
    nodes, node_idxs, node_attbs, r_discons, state_outer, n_discons, index_lists = \
            read_info_for_projection(dir_PM, dir_NM)

    # Do the projection. 
    vsh_projection_quick(dir_PM, dir_NM, l_max, eigvec_path_base, nodes, node_idxs, node_attbs, index_lists, r_discons, i_mode, save_spatial = save_spatial)

    return

def vsh_projection_quick_all_modes(dir_PM, dir_NM, l_max, eigvec_path_base, save_spatial = False):
    '''
    A wrapper for vsh_projection_quick(), which assembles the relevant information and loops over all of the modes.

    Input:

    dir_PM, dir_NM, l_max, eigvec_path_base, save_spatial
        See 'Definitions of variables'.

    Output:

    None (output is saved to file).
    '''

    # Define directory names.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Read various information about the mesh.
    nodes, node_idxs, node_attbs, r_discons, state_outer, n_discons, index_lists = \
            read_info_for_projection(dir_PM, dir_NM)

    # Get a list of modes.
    i_mode_list = get_list_of_modes_from_output_files(dir_NM)

    # Loop over the mode list.
    for i_mode in i_mode_list:
        
        print('\nProcessing mode: {:>5d}'.format(i_mode))

        # Do the projection. 
        vsh_projection_quick(dir_PM, dir_NM, l_max, eigvec_path_base, nodes, node_idxs, node_attbs, index_lists, r_discons, i_mode, save_spatial = save_spatial)

    return

def vsh_projection_quick_parallel(dir_PM, dir_NM, l_max, eigvec_path_base, save_spatial = False):
    '''
    A wrapper for vsh_projection_quick(), which assembles the relevant information and loops over all of the modes using all available processors.

    Input:

    dir_PM, dir_NM, l_max, eigvec_path_base, save_spatial
        See 'Definitions of variables'.

    Output:

    None (output is saved to file).
    '''

    # Define directory names.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Read various information about the mesh.
    nodes, node_idxs, node_attbs, r_discons, state_outer, n_discons, index_lists = \
            read_info_for_projection(dir_PM, dir_NM)

    # Get a list of modes.
    i_mode_list = get_list_of_modes_from_output_files(dir_NM)

    # Open a parallel pool.
    n_processes = multiprocessing.cpu_count()  
    print('Creating pool with {:d} processes.'.format(n_processes))
    with multiprocessing.Pool(processes = n_processes) as pool:
        
        # Use the pool to analyse the modes specified by num_span.
        # Note that the partial() function is used to meet the requirement of pool.map() of a pickleable function with a single input.
        pool.map(partial(vsh_projection_quick, dir_PM, dir_NM, l_max, eigvec_path_base, nodes, node_idxs, node_attbs, index_lists, r_discons, save_spatial = False), i_mode_list)

    return

# Main. -----------------------------------------------------------------------
def main():
    
    # Read the input file.
    input_file = 'input_NMPostProcess.txt'
    with open(input_file, 'r') as in_id:

        input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    input_args = [x.strip() for x in input_args]
    #
    dir_PM      = input_args[0]
    dir_NM      = input_args[1]
    option      = input_args[2]
    l_max       = int(input_args[3])
    i_mode_str  = input_args[4]

    # Do pre-processing steps.
    pre_process(dir_PM, dir_NM)
    eigvec_path_base = get_eigvec_path_base(dir_NM)

    # Quick projection (one radius only).
    if option == 'quick':
        
        # Loop over all modes.
        if i_mode_str == 'all':
            
            vsh_projection_quick_all_modes(dir_PM, dir_NM, l_max, eigvec_path_base)

        # Loop over all modes using all available processors.
        elif i_mode_str == 'parallel':

            vsh_projection_quick_parallel(dir_PM, dir_NM, l_max, eigvec_path_base)
        
        # Calculate a single mode.
        else:

            i_mode = int(i_mode_str)
            vsh_projection_quick_wrapper(dir_PM, dir_NM, l_max, i_mode, eigvec_path_base)

    else:

        print('Option {:} in file {:} not recognised.'.format(option, input_file))

    return

if __name__ == '__main__':

    main()
