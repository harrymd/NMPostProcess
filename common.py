'''
NMPostProcess/common.py
Various scripts used by multiple functions.
'''

# Import standard modules.
from glob import glob
import os

# Import third-party modules.
import numpy as np

# Generic operations. ---------------------------------------------------------
def mkdir_if_not_exist(dir_):
    '''
    Creates a directory if it does not exist.

    Input

    dir_    The path of the directory to be created.

    Output

            None
    '''

    if not os.path.exists(dir_):
        
        print('Creating directory {:}'.format(dir_))
        os.mkdir(dir_)

    else:

        print('Directory {:} exists. Skipping directory creation.'.format(dir_))

    return

# Read files.
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

def read_eigenvalues(file_eigval_list):
    '''
    Reads the eigenvalue list file generated by process.write_eigenvalues().

    Input

    file_eigval_list    The eigenvalue file.

    Output

    i_mode              A list of mode IDs (1, 2, ..., n_mode).
    freq                A list of frequencies.
    '''
    
    data    = np.loadtxt(file_eigval_list)
    num     = data[:, 0].astype(np.int)
    freq    = data[:, 1]
    
    return num, freq

def get_list_of_modes_from_regex(path_regex):

    path_list = glob(path_regex)
    #
    i_mode_list = []
    for path in path_list:

        # Remove .dat suffix.
        path = path[:-4]

        # Get integer at end of file name.
        try:

            i_mode = int(path.split('_')[-1])
        
            # Save.
            i_mode_list.append(i_mode)

        # Ignore other files e.g. *_vlist.dat which do not end with integers. 
        except ValueError:

            pass

    # Convert to NumPy array and sort.
    i_mode_list = np.array(i_mode_list, dtype = np.int)
    i_mode_list = np.sort(i_mode_list)

    return i_mode_list

def get_list_of_modes_from_output_files(dir_NM):
    '''
    Searches output directory for matching filenames and returns a list of mode IDs.

    Input:

    dir_NM
        See 'Definition of variables' in NMPostProcess/process.py.

    Output:

    i_mode_list
        A list of mode ID integers.
    '''

    regex_eigvec = '*.dat'
    path_regex_eigvec = os.path.join(dir_NM, regex_eigvec)
    i_mode_list = get_list_of_modes_from_regex(path_regex_eigvec)

    return i_mode_list

def get_list_of_modes_from_coeff_files(dir_NM, option):
    
    if option == 'quick':     

        regex_coeffs = 'quick_spectral_[0-9][0-9][0-9][0-9][0-9].npy' 

    elif option == 'full':

        regex_coeffs = 'full_spectral_[0-9][0-9][0-9][0-9][0-9].npy' 

    else:

        raise ValueError

    path_regex_coeffs = os.path.join(dir_NM, 'processed', 'spectral', regex_coeffs)
    i_mode_list = get_list_of_modes_from_regex(path_regex_coeffs) 

    return i_mode_list

def load_vsh_coefficients(dir_NM, i_mode, i_radius = None):
    '''
    Loads the vector spherical harmonic coefficients created by NMPostProcess/process.py.

    Input:

    dir_NM, i_mode
        See 'Definitions of variables'.

    Output:

    Ulm, Vlm, Wlm, scale, r_max, i_region_max
        See 'Definition of variables' in NMPostProcess/process.py.
    '''

    # Create the file path and load the NumPy array.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')

    # Infer whether the coefficients are 'quick' or 'full'
    if i_radius is None:

        option = 'quick'

    else:

        option = 'full'

    # Load the coefficients.
    # For 'quick' mode, there is a singleton dimension which can be
    # removed with 'squeeze'.
    file_spectral_data = '{:}_spectral_{:>05d}.npy'.format(option, i_mode)
    path_spectral_data = os.path.join(dir_spectral, file_spectral_data)
    coeffs = np.squeeze(np.load(path_spectral_data))

    # Load the header.
    file_header = '{:}_spectral_header_{:05d}.npy'.format(option, i_mode)
    path_header = os.path.join(dir_spectral, file_header)
    header = np.load(path_header) 
    n_radii = (len(header) - 3)//2
    header_info = dict()
    header_info['eigvec_max']   = header[0]
    header_info['r_max']        = header[1]
    header_info['i_region_max'] = int(header[2])
    header_info['r_sample']     = np.array(header[3              : 3 +   n_radii])
    header_info['i_sample']     = header[3 + n_radii    : 3 + 2*n_radii]
    header_info['i_sample']     = np.array(header_info['i_sample'], dtype = np.int)

    if option == 'full':

        if i_radius == 'all':

            return coeffs, header_info

        coeffs = coeffs[i_radius, ...]
        i_sample = header_info['i_sample'][i_radius]
        r_sample = header_info['r_sample'][i_radius]

    else:

        i_sample = header_info['i_sample'][0]
        r_sample = header_info['r_sample'][0]

    Ulm, Vlm, Wlm = coeffs

    return Ulm, Vlm, Wlm, r_sample, i_sample, header_info 

def read_input_NMPostProcess():

    # Read the input file.
    input_file = 'input_NMPostProcess.txt'
    with open(input_file, 'r') as in_id:

        input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    input_args = [x.strip().split() for x in input_args]
    #
    dir_PM      = input_args[0][0]
    dir_NM      = input_args[1][0]
    option      = input_args[2][0]
    if option == 'full':

        n_radii = int(input_args[2][1])
    
    else:

        n_radii = None

    l_max       = int(input_args[3][0])
    i_mode_str  = input_args[4][0]

    return dir_PM, dir_NM, option, l_max, i_mode_str, n_radii 

def read_input_plotting():

    # Read the plotting input file.
    plot_input_file = 'input_plotting.txt'
    with open(plot_input_file, 'r') as in_id:

        plot_input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    plot_input_args = [x.strip().split() for x in plot_input_args]
    option          = plot_input_args[0][0] 
    if option == 'full':
        
        i_radius_str = plot_input_args[0][1]
    
    else:

        i_radius_str = None

    plot_type       = plot_input_args[1][0]
    if plot_type == 'spatial':
        
        n_lat_grid = int(plot_input_args[1][1])

    else:

        n_lat_grid = None

    i_mode_str      = plot_input_args[2][0]
    fmt             = plot_input_args[3][0]

    return option, i_radius_str, plot_type, i_mode_str, fmt, n_lat_grid

# Functions related to spherical harmonics.
def convert_complex_sh_to_real(Xlm, l_max):
    '''
    Converts complex spherical harmonics to real ones.
    Dahlen and Tromp (1998), eq. B.98.

    Input:

    Xlm
        The complex spherical harmonic coefficients in SHTns format. The number of coefficients is (l_max + 1)(l_max + 2)/2.
    l_max
        The maximum angular order which was used in SHTns to calculate the coefficients.
    
    Returns:

    xlm
        The real spherical harmonic coefficients. The number of coefficients is (l_max + 1)*(l_max + 1).
    l_real
        A list of the l-values corresponding to each coefficient.
    m_real
        A list of the m-values corresponding to each coefficient.
    '''
    
    # Initialise output arrays.
    sqrt2 = np.sqrt(2.0)
    n_coeff = (l_max + 1)**2
    xlm     = np.zeros(n_coeff)
    l_real  = np.zeros(n_coeff, dtype = np.int)
    m_real  = np.zeros(n_coeff, dtype = np.int) 
    
    i_complex = 0
    i_real = 0
    for m_complex in range(l_max + 1):

        for l_complex in range(m_complex, l_max + 1):
            
            coeff_complex = Xlm[i_complex]
            
            if m_complex == 0:
                
                assert np.imag(coeff_complex) == 0.0
                
                i_real_0 = l_complex*(l_complex + 1)

                xlm[i_real_0] = np.real(coeff_complex)

                m_real[i_real_0] = 0
                l_real[i_real_0] = l_complex

            else:
                
                i_real_pos = (l_complex*(l_complex + 1)) + m_complex
                i_real_neg = (l_complex*(l_complex + 1)) - m_complex

                xlm[i_real_pos] =  sqrt2*np.real(coeff_complex)
                xlm[i_real_neg] = -sqrt2*np.imag(coeff_complex)

                m_real[i_real_pos] =  m_complex
                m_real[i_real_neg] = -m_complex

                l_real[i_real_pos] = l_complex
                l_real[i_real_neg] = l_complex

            i_complex = i_complex + 1

    return xlm, l_real, m_real

def make_l_and_m_lists(l_max):
    '''
    Calculates the lists of the angular and azimuthal order of coefficients stored in SHTns format.

    Input:

    l_max
        The maximum l-value used for calculation the coefficients with SHTns.

    Output:

    l_list, m_list
        The l- and m-values corresponding to each coefficient.
    '''

    n = (l_max + 1)*(l_max + 2)//2
    
    l_list = np.zeros(n, dtype = np.int)
    m_list = np.zeros(n, dtype = np.int)

    k = 0
    for i in range(l_max + 1):
        
        for j in range(i, l_max + 1):
            
            l_list[k] = j
            m_list[k] = i

            k = k + 1

    return l_list, m_list

# Miscellaneous functions. ----------------------------------------------------
def lf_clusters(l, f, f_tol = 0.001):
    '''
    Groups modes with the same l-value and similar frequencies into clusters.
    
    Variables

    (n_modes)   The number of modes.

    Input

    l       (n_modes) A list of the l-value of each mode.
    f       (n_modes) A list of the frequency of each mode.
    f_tol   The maximum frequency gap within a cluster.

    Output

    i_clusters      (n_clusters) For each cluster, a list (of length cluster_multipliticies[i]) of the indices of the modes of the cluster within the original list.
    l_clusters      (n_clusters) For each cluster, the l-value of that cluster.
    f_clusters      (n_clusters) For each cluster, a list (of length cluster_multiplicities[i]) of the frequencies of the modes of the cluster. 
    n_clusters      The number of clusters found.
    f_cluster_means (n_clusters) The mean frequency of each cluster.
    cluster_multiplicities
                    (n_cluster) The number of modes in each cluster.
    '''

    # Initialise output arrays.
    l_clusters = []
    f_clusters = []
    i_clusters = []

    # Initialise loop variables.
    # i_remain  A list of indices of modes which have not yet been assigned a cluster.
    # n_remain  The number of modes which have not been assigned a cluster.
    n_modes = len(l)
    i_remain = list(range(0, n_modes))
    n_remain = len(i_remain)

    # Assign each mode a cluster.
    while n_remain > 0:
        
        # Start a new cluster.
        l_cluster = l[i_remain[0]]
        #
        i_cluster = [i_remain[0]]
        f_cluster = [f[i_remain[0]]]
        #
        del i_remain[0]
        n_remain = len(i_remain)
        
        j = 0
        while j < n_remain:
            
            i = i_remain[j]

            # Find the bounds of the current cluster.
            f_min = np.min(f_cluster)
            f_max = np.max(f_cluster)
            
            # Add modes with matching l-values, but only if they
            # are close (in frequency) to the current cluster.
            if l[i] == l_cluster:
                
                # Check if the mode frequency is close enough to the
                # current cluster.
                if (f[i] > (f_min - f_tol)) and (f[i] < (f_max + f_tol)):

                    # Add the mode to the cluster.
                    i_cluster.append(i)
                    f_cluster.append(f[i])

                    del i_remain[j]
                    n_remain = len(i_remain)
                    
                    # Restart the search for this value, as the
                    # cluster has a new entry and its bounding 
                    # frequencies may have changed.
                    j = 0
                    continue

            j = j + 1
    
        # Once all the l-values have been checked, the cluster is complete.
        # Store the cluster and its frequencies in the master list.
        l_clusters.append(l_cluster)
        f_clusters.append(np.array(f_cluster))
        i_clusters.append(np.array(i_cluster, dtype = np.int))

    # Find how many clusters there are.
    n_clusters = len(l_clusters)

    # Find the mean frequency of each cluster.
    f_cluster_means = np.zeros(n_clusters)
    for i in range(n_clusters):

        f_cluster_means[i] = np.mean(f_clusters[i])
    
    # Find the number of modes in each cluster.
    cluster_multiplicities = np.zeros(n_clusters, dtype = np.int32)
    for i in range(n_clusters):

        cluster_multiplicities[i] = len(f_clusters[i])

    return i_clusters, l_clusters, f_clusters, n_clusters, f_cluster_means, cluster_multiplicities
