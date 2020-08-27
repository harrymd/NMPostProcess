'''
NMPostProcess/plot/plot_mode_diagram.py
Plots a mode diagram based on vector spherical harmonic expansion.
'''

# Load modules. ---------------------------------------------------------------

# Load standard modules.
import os

# Load third-party modules.
import matplotlib.pyplot as plt
import numpy as np

# Load local modules.
from common import lf_clusters, read_eigenvalues

# Utilities. ------------------------------------------------------------------
def label_multiplets(ax, cluster_multiplicities, l_clusters, f_cluster_means, missing_modes = None):
    '''
    Labels multiplets in the mode diagram.

    Input:

    ax
        The axis upon which to put the labels.
    cluster_multiplicities, l_clusters, f_cluster_means
        See lf_clusters() in NMPostProcess/common.py.
    
    Output:

    None
        
    '''
    
    # mode = 'excess' will label the clusters with too many modes (for their given l-value) in green, and those with too few in red. Other modes will not be labelled.
    # mode = 'all' will label each cluster with the number of modes in contains (in black).
    mode = 'excess'

    n_clusters = len(l_clusters)

    # Label the multiplets.
    for i in range(n_clusters):
        
        if mode is 'excess':
             
            excess = cluster_multiplicities[i] - ((2*l_clusters[i]) + 1)
            if excess == 0:
                
                skip = True
            
            else: 

                skip = False
                font_size = 8

                if excess > 0:

                    font_colour = 'g'
                    label = excess

                else:

                    font_colour = 'r'
                    label = abs(excess)

        else:

            skip = False
            label = cluster_multiplicities[i]

            if cluster_multiplicities[i] == ((2*l_clusters[i]) + 1):
                
                font_size = 8
                font_colour = 'g'

            else:
                
                font_size = 10
                font_colour = 'r'
        
        if not skip:

            ax.annotate('{:d}'.format(label),
                    [l_clusters[i], f_cluster_means[i]],
                    [5, -5],
                    textcoords  = 'offset points',
                    color       = font_colour,
                    size        = font_size)
            
    if missing_modes is not None:
        
        if mode is 'excess':
            
            font_color = 'r'
            font_size = 8

            for missing_mode in missing_modes:

                l, f = missing_mode
                label = 2*l + 1

                ax.annotate('{:d}'.format(label),
                        [l, f],
                        [5, -5],
                        textcoords  = 'offset points',
                        color       = font_color,
                        size        = font_size)

        else:

            raise NotImplementedError

# Plotting. -------------------------------------------------------------------
def plot_mode_diagram_core(mode_info, ax = None, show = True, label_clusters = True, path_fig = None):
    '''
    Plots angular order versus frequency.

    Input

    mode_info   A dictionary. Keys are mode types, e.g. 'R', 'S' or 'T0'. Each mode type stores a dictionary with keys 'l' and 'f' which store the angular order and frequency of the modes of each type. 

    ax          (Optional.) Axis upon which plot is drawn.
    show        (Optional.) If True, show the plot when finished.

    Output

    None
    '''

    # Create axes (if none are supplied).
    if ax is None:

        fig = plt.figure(figsize = (5.0, 5.0))
        ax = plt.gca()

    # Loop over the different kinds of mode.
    for type_ in mode_info:

        # Plot a point (l, f) for each mode of this type.
        ax.scatter(mode_info[type_]['l'], mode_info[type_]['f'], label = type_)

    # Label mode clusters.
    if label_clusters:

        f_max = np.max([np.max(mode_info[type_]['f']) for type_ in mode_info])
        f_tol = 0.01*f_max

        for type_ in mode_info:

            i_clusters, l_clusters, f_clusters, n_clusters, f_cluster_means, \
            cluster_multiplicities = lf_clusters(mode_info[type_]['l'], mode_info[type_]['f'], f_tol = f_tol)

            label_multiplets(ax, cluster_multiplicities, l_clusters, f_cluster_means, missing_modes = None)

    # Add the legend.
    plt.legend()

    # Axis labels.
    ax.set_xlabel('Angular order, $\ell$')
    ax.set_ylabel('Frequency (mHz)')

    # Save figure (if requested).
    if path_fig is not None:

        print('Saving figure to {:}'.format(path_fig))
        plt.savefig(path_fig, dpi = 300)

    # Show the plot (if requested).
    if show:

        plt.show()

    return

def plot_mode_diagram_wrapper(dir_NM):
    '''
    Reads mode information files and plots angular order versus frequency.

    Input:

    dir_NM  See 'Definitions of variables' in process.py.
    
    Output:

    None
    '''

    # Define directories and files.
    dir_processed = os.path.join(dir_NM, 'processed')
    path_fig = os.path.join(dir_processed, 'plots', 'mode_diagram.png')

    # Load the mode identification information.
    path_ids = os.path.join(dir_processed, 'mode_ids_quick.txt')
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)

    # Group the modes.
    type_str_list = ['R', 'S', 'T']
    n_modes = len(i_mode)
    mode_info = dict()
    for i in range(n_modes):

        # Get string ('R', 'S', or 'T') of mode type.
        type_str = type_str_list[type_[i]]

        if type_str == 'T':

            # Toroidal modes have an integer indicating the shell they belong to.
            type_str = 'T{:>1d}'.format(shell[i])

        # Create a new entry in the mode dictionary if no modes of this type have been recorded.
        if type_str not in mode_info.keys():

            # Store the information for this mode.
            mode_info[type_str] = dict()
            mode_info[type_str]['l'] = np.atleast_1d(l[i])
            mode_info[type_str]['f'] = np.atleast_1d(f[i])

        # Otherwise, add the mode information to the existing array for this mode type.
        else:

            # Store the information for this mode.
            mode_info[type_str]['l'] = np.append(mode_info[type_str]['l'], l[i])
            mode_info[type_str]['f'] = np.append(mode_info[type_str]['f'], f[i])

    # Plot.
    plot_mode_diagram_core(mode_info, path_fig = path_fig)

    return

def main():

    # Read the NMPostProcess input file.
    input_file = 'input_NMPostProcess.txt'
    with open(input_file, 'r') as in_id:

        input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    input_args = [x.strip() for x in input_args]
    #dir_PM      = input_args[0]
    dir_NM      = input_args[1]
    #option      = input_args[2]
    #l_max       = int(input_args[3])

    plot_mode_diagram_wrapper(dir_NM)

if __name__ == '__main__':

    main()
