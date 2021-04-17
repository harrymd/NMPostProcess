import argparse
import os

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from NMPostProcess.common import (mkdir_if_not_exist, mode_id_information_to_dict, 
                            read_eigenvalues, read_input_NMPostProcess)

def plot_freq_wrapper(dir_NM, option, name_mode_list):

    # Define directories and files.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)
    path_fig = os.path.join(dir_plot, 'freq_diagram_{:}.png'.format(name_mode_list))

    # Load the mode identification information.
    path_ids = os.path.join(dir_processed, 'mode_ids_{:}.txt'.format(option))
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)

    ## Group the modes.
    #mode_info = mode_id_information_to_dict(type_, l, f, shell)
    
    # Read the mode list.
    dir_mode_lists = os.path.join(dir_processed, 'mode_lists')
    path_mode_list = os.path.join(dir_mode_lists, '{:}.txt'.format(name_mode_list))
    print('Reading {:}'.format(path_mode_list))
    mode_list = np.loadtxt(path_mode_list, dtype = np.int)
    
    # Get the frequencies.
    f_list = f[mode_list - 1] # Note offset between 0- and 1-based indexing.
    
    # Plot.
    plot_freq_core([f_list], [''], path_out = path_fig)

    return

def plot_freq_core(f_lists, label_list, path_out = None, vertical = False, tight_layout = True, show = True, ax = None, size_f_axis = 10.0, size_other_axis = 2.0):
    
    if ax is None:

        if vertical:
        
            figsize = (size_other_axis, size_f_axis)

        else:

            figsize = (size_f_axis, size_other_axis)

        fig = plt.figure(figsize = figsize)
        ax = plt.gca()
    
    f_list = []
    for f_list_i in f_lists:
        f_list.extend(f_list_i)

    f_min = np.min(f_list)
    f_max = np.max(f_list)
    f_range = f_max - f_min
    frac_pad = 0.1

    font_size_label = 12

    f_lims = [f_min - frac_pad*f_range, f_max + frac_pad*f_range]
    
    n_clusters = len(f_lists)

    for i in range(n_clusters):
        
        f = f_lists[i]
        f_mean = np.mean(f)
        f_min = np.min(f)
        n_modes = len(f)
    
        segments = []
        for j in range(n_modes):
            
            if vertical:

                segment = [[0.0, f[j]], [0.6, f[j]]]
            
            else:
                
                segment = [[f[j], 0], [f[j], 0.6]]

            segments.append(segment)

        #line_coll = LineCollection(segments, linewidths = 0.5, colors = 'k', alpha = 0.3)
        line_coll = LineCollection(segments, linewidths = 1.5, colors = 'k', alpha = 0.5)

        ax.add_collection(line_coll) 
        
        #label_str = '$_{{{:d}}}${:}$_{{{:d}}}$'.format(*clusters[i])
        label_str = label_list[i]
        
        if vertical:

            ax.text(0.3, f_min - 0.005, label_str, ha = 'center', va = 'top', transform = ax.transData, fontsize = font_size_label)

        else:

            ax.text(f_mean, 0.8, label_str, ha = 'center', va = 'center', transform = ax.transData, fontsize = font_size_label)
    
    if vertical:
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim(f_lims)

        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axes.get_xaxis().set_ticks([])

        ax.set_ylabel('Frequency / mHz', fontsize = font_size_label)

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    else:

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axes.get_yaxis().set_ticks([])

        ax.set_xlim(f_lims)
        ax.set_ylim([0.0, 1.0])

        ax.set_xlabel('Frequency / mHz', fontsize = font_size_label)
    
    if tight_layout:

        plt.tight_layout()

    if path_out is not None:
        
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.0)

        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.0)

        print('Saving figure to {:}'.format(path_out))
        plt.savefig(path_out, dpi = 300, bbox_inches = 'tight')
    
    if show:

        plt.show()

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("name_mode_list", help = "Name of file listing modes.")
    args = parser.parse_args()
    
    # Rename input arguments.
    name_mode_list = args.name_mode_list

    # Read the NMPostProcess input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii = read_input_NMPostProcess()

    # Plot.
    plot_freq_wrapper(dir_NM, option, name_mode_list)

    return

if __name__ == '__main__':

    main()
