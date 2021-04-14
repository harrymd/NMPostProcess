'''
NMPostProcess/plot/plot_mode_diagram.py
Plots a mode diagram based on vector spherical harmonic expansion.
'''

# Load modules. ---------------------------------------------------------------

# Load standard modules.
import argparse
import os

# Load third-party modules.
import matplotlib.pyplot as plt
import numpy as np

# Load local modules.
from common import mkdir_if_not_exist, lf_clusters, read_eigenvalues, read_input_NMPostProcess, reference_mode_info_to_dict, mode_id_information_to_dict 

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
        
        if mode == 'excess':
             
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
        
        if mode == 'excess':
            
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
def plot_mode_diagram_core(mode_info, ax = None, show = True, label_clusters = True, path_fig = None, nlf_ref = None, cluster_tol = None, interactive = False):
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

        fig = plt.figure(figsize = (8.0, 6.0), tight_layout = True)
        ax = plt.gca()

    picker = 3 

    # Loop over the different kinds of mode.
    c_dict = {'R' : 'red', 'S' : 'blue', 'T0' : 'orange', 'T2' : 'green'}
    scatter_handle_dict = dict()
    for type_ in mode_info:

        if type_ in c_dict.keys():

            c = c_dict[type_]

        else:

            c = None

        # Plot a point (l, f) for each mode of this type.
        scatter_handle = ax.scatter(mode_info[type_]['l'], mode_info[type_]['f'], label = type_,
                    c = c, picker = picker)

        scatter_handle_dict[type_] = scatter_handle

    # Label mode clusters.
    if label_clusters:

        f_max = np.max([np.max(mode_info[type_]['f']) for type_ in mode_info])

        if cluster_tol is None:

            f_tol = 0.02*f_max

        else:

            f_tol = cluster_tol
            
        print('Cluster tolerance: {:.3f} mHz'.format(f_tol))

        for type_ in mode_info:

            i_clusters, l_clusters, f_clusters, n_clusters, f_cluster_means, \
            cluster_multiplicities = lf_clusters(mode_info[type_]['l'], mode_info[type_]['f'], f_tol = f_tol)

            label_multiplets(ax, cluster_multiplicities, l_clusters, f_cluster_means, missing_modes = None)

    # Plot reference dispersion.
    if nlf_ref is not None:

        for mode_type in nlf_ref:

            if nlf_ref[mode_type] is not None:

                mode_info_ref = nlf_ref[mode_type]
                n_ref = mode_info_ref['n']
                l_ref = mode_info_ref['l']
                f_ref = mode_info_ref['f']

                n_list = np.sort(np.unique(n_ref))

                for n_i in n_list:

                    i = np.where(n_ref == n_i)

                    ax.plot(l_ref[i], f_ref[i], c = c_dict[mode_type])
                    ax.scatter(l_ref[i], f_ref[i], c = 'k')

    # Add the legend.
    plt.legend()

    # Axis labels.
    fontsize_label = 12
    ax.set_xlabel('Angular order, $\ell$', fontsize = fontsize_label)
    ax.set_ylabel('Frequency (mHz)', fontsize = fontsize_label)

    # Determine axes limits.
    l_min = 9999999
    l_max = 0
    f_min = np.inf
    f_max = 0.0
    for mode_type in mode_info:

        f_min = np.min([f_min, np.min(mode_info[mode_type]['f'])])
        f_max = np.max([f_max, np.max(mode_info[mode_type]['f'])])

        l_min = np.min([l_min, np.min(mode_info[mode_type]['l'])])
        l_max = np.max([l_max, np.max(mode_info[mode_type]['l'])])

    l_range = l_max - l_min
    f_range = f_max - f_min
    buff = 0.05

    l_lim_min = l_min - l_range*buff
    if l_lim_min < 0:
        
        l_lim_min = 0

    l_lim_max = l_max + l_range*buff

    f_lim_min = f_min - f_range*buff
    if f_lim_min < 0.0:
        
        f_lim_min = 0.0

    f_lim_max = f_max + f_range*buff

    #ax.set_xlim([0.0, 32.0])
    #ax.set_ylim([2.9, 3.8])
    ax.set_xlim([l_lim_min, l_lim_max])
    ax.set_ylim([f_lim_min, f_lim_max])

    # Save figure (if requested).
    if path_fig is not None:

        print('Saving figure to {:}'.format(path_fig))
        plt.savefig(path_fig, dpi = 300)

    if interactive:

        print("Interactive mode.")

        #ax.text(0.9, 0.9, "Mode selected:",
        #        transform = ax.transAxes,
        #        ha = 'right',
        #        va = 'top')
        
        ev_mode_type = "NA"
        ev_mode_l    = 0
        ev_mode_f    = 0.0
        ev_mode_i    = 0

        ev_mode_fmt = "Type: {:>2}\n$\ell$: {:>2d}\n$f$: {:>7.4f} mHz\nIndex: {:>4d}"
        ev_mode_label_str = ev_mode_fmt.format(ev_mode_type, ev_mode_l, ev_mode_f, ev_mode_i)
        ev_mode_label = ax.text(0.5, 0.85, ev_mode_label_str,
                            transform = ax.transAxes,
                            ha = 'left',
                            va = 'top')

        ev_mode_marker = ax.scatter([], [], marker = '+', zorder = 20, c = 'g', s = 20)

        def on_pick(event):

            ind = event.ind[0]
            artist = event.artist

            ev_mode_type = list(scatter_handle_dict.keys())[list(scatter_handle_dict.values()).index(artist)]

            ev_mode_l = mode_info[ev_mode_type]['l'][ind]
            ev_mode_f = mode_info[ev_mode_type]['f'][ind]
            ev_mode_i = mode_info[ev_mode_type]['i'][ind]
            #data_subset = scatter_handles.index(artist)
            #data_value = data_subset[ind]
            #print(data_value)
            #print("Artist picked: {:}".format(artist))
            
            ev_mode_label_str = ev_mode_fmt.format(ev_mode_type, ev_mode_l, ev_mode_f, ev_mode_i)
            ev_mode_label.set_text(ev_mode_label_str)
            #print(ev_mode_type)

            ev_mode_marker.set_offsets([ev_mode_l, ev_mode_f])

            plt.draw()

            return

        fig.canvas.callbacks.connect('pick_event', on_pick)

    # Show the plot (if requested).
    if show:

        plt.show()

    return

def plot_mode_diagram_wrapper(dir_NM, option, paths_ref = None, cluster_tol = None, interactive = False):
    '''
    Reads mode information files and plots angular order versus frequency.

    Input:

    dir_NM  See 'Definitions of variables' in process.py.
    
    Output:

    None
    '''

    # Define directories and files.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)
    path_fig = os.path.join(dir_plot, 'mode_diagram_{:}.png'.format(option))

    # Load the mode identification information.
    path_ids = os.path.join(dir_processed, 'mode_ids_{:}.txt'.format(option))
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)

    # Group the modes.
    mode_info = mode_id_information_to_dict(type_, l, f, shell)
    #type_str_list = ['R', 'S', 'T']
    #n_modes = len(i_mode)
    #mode_info = dict()
    #for i in range(n_modes):

    #    # Get string ('R', 'S', or 'T') of mode type.
    #    type_str = type_str_list[type_[i]]

    #    if type_str == 'T':

    #        # Toroidal modes have an integer indicating the shell they belong to.
    #        type_str = 'T{:>1d}'.format(shell[i])

    #    # Create a new entry in the mode dictionary if no modes of this type have been recorded.
    #    if type_str not in mode_info.keys():

    #        # Store the information for this mode.
    #        mode_info[type_str] = dict()
    #        mode_info[type_str]['l'] = np.atleast_1d(l[i])
    #        mode_info[type_str]['f'] = np.atleast_1d(f[i])

    #    # Otherwise, add the mode information to the existing array for this mode type.
    #    else:

    #        # Store the information for this mode.
    #        mode_info[type_str]['l'] = np.append(mode_info[type_str]['l'], l[i])
    #        mode_info[type_str]['f'] = np.append(mode_info[type_str]['f'], f[i])
    
    # Load reference dispersion diagram (if available).
    if paths_ref is not None:

        nlf_ref = reference_mode_info_to_dict(paths_ref)

    #    nlf_ref = dict()
    #    for mode_type in paths_ref:
    #        
    #        if paths_ref[mode_type] is not None:

    #            nlf_ref[mode_type] = dict()
    #            nlf_ref_i = np.loadtxt(paths_ref[mode_type])
    #            nlf_ref[mode_type]['n'] = nlf_ref_i[:, 0].astype(np.int)
    #            nlf_ref[mode_type]['l'] = nlf_ref_i[:, 1].astype(np.int)
    #            nlf_ref[mode_type]['f'] = nlf_ref_i[:, 2]

    #        else:

    #            nlf_ref[mode_type] = None

    else:

        nlf_ref = None

    # Plot.
    plot_mode_diagram_core(mode_info,
            path_fig = path_fig,
            nlf_ref = nlf_ref,
            cluster_tol = cluster_tol,
            interactive = interactive)

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', action = 'append', nargs = 2,
    metavar=('mode_type', 'path'), help = "Path to a file containing mode information (n, l, and frequency (mHz)) for the specified mode type (e.g. S) for a reference model, to add to the plot."),
    parser.add_argument('--cluster_tol', type = float)
    parser.add_argument('--interactive', action = 'store_true', help = "Include this flag to allow interaction with plot.")
    input_args = parser.parse_args()
    cluster_tol = input_args.cluster_tol
    interactive = input_args.interactive

    # Parse reference paths.
    if input_args.ref_path is None:

        paths_ref = None

    else:

        paths_ref = dict()
        for i in range(len(input_args.ref_path)):
            
            mode_type = input_args.ref_path[i][0]
            path_ref = input_args.ref_path[i][1]
            paths_ref[mode_type] = path_ref

    # Read the NMPostProcess input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii = read_input_NMPostProcess()

    # Plot the mode diagram.
    plot_mode_diagram_wrapper(dir_NM, option,
            paths_ref = paths_ref,
            cluster_tol = cluster_tol,
            interactive = interactive)

if __name__ == '__main__':

    main()
