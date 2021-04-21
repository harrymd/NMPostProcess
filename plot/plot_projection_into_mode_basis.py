import argparse
from operator import sub
import os

from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform
import numpy as np

from NMPostProcess.common import (get_list_of_modes_from_coeff_files,
                        read_eigenvalues, read_input_NMPostProcess)
from NMPostProcess.project_into_mode_basis import load_1d_mode_info

def load_3d_mode_info(path_in):

    # Dictionary for loading mode type saved as integer.
    mode_type_from_int_dict = {0 : 'R', 1 : 'S', 2 : 'T1', 3 : 'T0'}

    # Load data and extract columns.
    data = np.loadtxt(path_in)
    mode_type_int   = data[:, 0].astype(np.int)
    n               = data[:, 1].astype(np.int)
    l               = data[:, 2].astype(np.int)
    m               = data[:, 3].astype(np.int)
    coeff_real      = data[:, 4]
    coeff_imag      = data[:, 5]

    # Separate different mode types.
    mode_info = dict()
    mode_type_int_list = np.sort(np.unique(mode_type_int))
    for mode_type_int_i in mode_type_int_list:

        # Get mode type string.
        mode_type = mode_type_from_int_dict[mode_type_int_i]
        mode_info[mode_type] = dict()

        # Find modes with the given mode type.
        j = np.where(mode_type_int_i == mode_type_int)[0]
        
        # Store select modes in dictionary.
        mode_info[mode_type]['n'] = n[j]
        mode_info[mode_type]['l'] = l[j]
        mode_info[mode_type]['m'] = m[j]
        mode_info[mode_type]['coeff_real'] = coeff_real[j]
        mode_info[mode_type]['coeff_imag'] = coeff_imag[j]

    # Group modes into specific multiplets.
    mode_info_grouped = dict()
    for mode_type in mode_info.keys():

        num_modes = len(mode_info[mode_type]['n'])

        mode_info_grouped[mode_type] = dict()

        for i in range(num_modes):

            # Unpack dictionary.
            n = mode_info[mode_type]['n'][i]
            l = mode_info[mode_type]['l'][i]
            #m = mode_info[mode_type]['m'][i]
            #coeff_real = mode_info[mode_type][coeff_real][i]
            #coeff_imag = mode_info[mode_type][coeff_imag][i]

            # Get unique string identifying multiplet.
            nl_key = '{:>05d}_{:>05d}'.format(n, l)

            if nl_key in mode_info_grouped[mode_type].keys():
                
                for variable in ['m', 'coeff_real', 'coeff_imag']:

                    mode_info_grouped[mode_type][nl_key][variable] = \
                        np.append(mode_info_grouped[mode_type][nl_key][variable],
                                    mode_info[mode_type][variable][i])

            else:

                mode_info_grouped[mode_type][nl_key] = dict()
                mode_info_grouped[mode_type][nl_key]['n'] = n
                mode_info_grouped[mode_type][nl_key]['l'] = l 

                for variable in ['m', 'coeff_real', 'coeff_imag']:

                    mode_info_grouped[mode_type][nl_key][variable] = \
                            np.atleast_1d(mode_info[mode_type][variable][i])

    # Sort the multiplets by m-value.
    for mode_type in mode_info_grouped.keys():

        for nl_key in mode_info_grouped[mode_type]:

            i_sort = np.argsort(mode_info_grouped[mode_type][nl_key]['m'])
            for variable in ['m', 'coeff_real', 'coeff_imag']:

                mode_info_grouped[mode_type][nl_key][variable] = \
                        mode_info_grouped[mode_type][nl_key][variable][i_sort]

    # Fill in empty mode types.
    for mode_type in ['R', 'S', 'T0', 'T1']:

        if mode_type not in mode_info_grouped.keys():

            mode_info_grouped[mode_type] = None

    #return mode_info
    return mode_info_grouped

def get_max_l(mode_info, f_lims):

    l_max = 0

    for mode_type in mode_info:
        
        f = mode_info[mode_type]['f']
        l = mode_info[mode_type]['l']

        i_freq = np.where((f < f_lims[1]) & (f > f_lims[0]))[0]

        l_max_new = np.max(l[i_freq])
        if l_max_new > l_max:

            l_max = l_max_new

    return l_max

def plot_dispersion(ax, mode_type, mode_info, color = 'k'):

    # Unpack.
    n = mode_info['n']
    l = mode_info['l']
    f = mode_info['f']

    # Get list of n values.
    n_unique = np.sort(np.unique(n))
    num_n_unique = len(n_unique)

    # Loop over n values.
    for i in range(num_n_unique):
        
        n_i = n_unique[i]
        j = np.where(n == n_i)[0]
        
        if mode_type != 'R':

            ax.plot(l[j], f[j], color = color)

        ax.scatter(l[j], f[j], color = color)

    return

def plot_excitation_wrapper(ax, mode_type, mode_info_1d, mode_info_3d, max_amp, custom_offsets = False, colorbar = True):
    
    if custom_offsets:

        multiplet_offset_dict = define_custom_offsets()
    
    else:
        
        multiplet_offset_dict = {key : None for key in mode_inf_3d.keys()}

    if mode_info_3d is not None:

        plot_excitation(ax, mode_info_1d, mode_info_3d, max_amp,
                multiplet_offset_dict = multiplet_offset_dict[mode_type],
                colorbar = colorbar)

    return

def define_custom_offsets():
    
    d_x = 0.01
    d_y = 0.01
    multiplet_offset_dict_R = {}
    multiplet_offset_dict_S = {
        '00000_00021' : [0.0, -d_y],
        '00004_00007' : [0.0, -d_y],
        '00002_00014' : [0.0, -d_y],
        '00000_00022' : [0.0, d_y/2.0],
        '00000_00023' : [0.0, -d_y],
        '00000_00025' : [0.0,  d_y],
        '00003_00012' : [0.0,  d_y],
        '00003_00013' : [0.0,  d_y],
        '00004_00008' : [0.0, -d_y],
        '00007_00004' : [0.0,  d_y],
        '00000_00026' : [d_x,  0.0],
        '00001_00017' : [0.0, -d_y],
        '00000_00028' : [0.0, -d_y],
        '00001_00018' : [0.0, -d_y],
        '00003_00014' : [0.0, d_y],
        '00002_00017' : [0.0,  2.0*d_y],
        }
    multiplet_offset_dict_T0 = {
        '00000_00006' : [0.0,  d_y/2.0],
                            }
    multiplet_offset_dict_T1 = {
        '00001_00013' : [0.0,  d_y],
        '00003_00003' : [0.0,  d_y],
        '00003_00004' : [0.0,  -d_y],
                                }

    multiplet_offset_dict = {   'R' : multiplet_offset_dict_R, 
                                'S' : multiplet_offset_dict_S,
                                'T0' : multiplet_offset_dict_T0,
                                'T1' : multiplet_offset_dict_T1 }

    return multiplet_offset_dict

def old_plot_excitation_ST(ax, mode_info_1d, mode_info_3d):

    fig = plt.gcf()
    size = fig.get_size_inches()*fig.dpi
    ratio = size[1]/size[0]

    # Count number of modes.
    num_modes_3d = 0
    for multiplet in mode_info_3d.keys():
        
        l = mode_info_3d[multiplet]['l']
        num_modes_3d = num_modes_3d + ((2 * l) + 1)

    # Prepare arrays containing scatter plot information.
    x_coord = np.zeros(num_modes_3d)
    y_coord = np.zeros(num_modes_3d)

    # Assign coordinates.
    i0 = 0
    radius_circle = 30 # pixel coordinates
    for multiplet in mode_info_3d.keys():

        # Unpack.
        n_3d = mode_info_3d[multiplet]['n']
        l_3d = mode_info_3d[multiplet]['l']

        # Create a circle of points (pixel coordinates), one per singlet.
        num_pts = (2 * l_3d) + 1
        theta_pts = np.linspace(0.0, 2.0 * np.pi, num = num_pts)
        x_pts = ratio * radius_circle * np.sin(theta_pts)
        y_pts = (1.0/ratio) * radius_circle * np.cos(theta_pts)

        # Get end index for output array.
        i1 = i0 + num_pts

        # Find frequency and angular frequency of mode for centering on the
        # mode diagram.
        condition = ((mode_info_1d['n'] == n_3d) &
                     (mode_info_1d['l'] == l_3d))
        assert np.sum(condition) == 1
        i_match = np.where(condition)[0][0]
        #
        n = mode_info_1d['n'][i_match]
        l = mode_info_1d['l'][i_match]
        f = mode_info_1d['f'][i_match]

        # Convert center coordinates to pixel coordinates.
        x0, y0 = ax.transData.transform((l, f))
        #print('{:>3d} {:>3d} {:>.3f} {:>10.1f} {:>10.1f}'.format(n, l, f, x0, y0))
        #print(ax.transData.inverted().transform((x0, y0)))

        # Apply offset.
        x_pts = x_pts + x0
        y_pts = y_pts + y0

        # Store.
        x_coord[i0 : i1] = x_pts
        y_coord[i0 : i1] = y_pts

        # Prepare for next iteration.
        i0 = i1

    #for i in range(num_modes_3d):

    #    x_coord[i], y_coord[i] = ax.transData.inverted().transform((x_coord[i], y_coord[i]))
    #x_coord, y_coord = ax.transData.inverted().transform((x_coord, y_coord)).T

    ax.scatter(x_coord, y_coord, color = 'k', s = 3)



    return

def plot_excitation(ax, mode_info_1d, mode_info_3d, max_amp, multiplet_offset_dict = None, colorbar = True):

    # Define color map.
    #c_map_name = 'cividis'
    #c_map_name = 'viridis'
    #c_map_name = 'viridis_r'
    c_map_name = 'magma_r'
    #c_map_name = 'Reds'
    #c_map = get_cmap(c_map_name)
    #c_map.set_bad(color = (0.0, 0.0, 0.0, 0.0))
    c_map = c_map_name
    c_norm = Normalize(vmin = 0.0, vmax = 1.0)

    # Count number of modes.
    num_modes_3d = 0
    for multiplet in mode_info_3d.keys():
        
        l = mode_info_3d[multiplet]['l']
        num_modes_3d = num_modes_3d + ((2 * l) + 1)

    # Prepare arrays containing scatter plot information.
    x_coord = np.zeros(num_modes_3d)
    y_coord = np.zeros(num_modes_3d)

    # Assign coordinates.
    i0 = 0
    for multiplet in mode_info_3d.keys():

        # Unpack.
        n_3d = mode_info_3d[multiplet]['n']
        l_3d = mode_info_3d[multiplet]['l']
        m_3d = mode_info_3d[multiplet]['m']
        coeff_real_3d = mode_info_3d[multiplet]['coeff_real']
        coeff_imag_3d = mode_info_3d[multiplet]['coeff_imag']
        coeff_amp = np.sqrt(coeff_real_3d**2.0 + coeff_imag_3d**2.0)
        coeff_amp_normalised = (coeff_amp / max_amp)
        
        # Get end index for output array.
        num_pts = (2 * l_3d) + 1
        i1 = i0 + num_pts

        # Find frequency and angular frequency of mode for centering on the
        # mode diagram.
        condition = ((mode_info_1d['n'] == n_3d) &
                     (mode_info_1d['l'] == l_3d))
        assert np.sum(condition) == 1
        i_match = np.where(condition)[0][0]
        #
        n = mode_info_1d['n'][i_match]
        l = mode_info_1d['l'][i_match]
        f = mode_info_1d['f'][i_match]

        # Create axis.
        fig = plt.gcf()
        fig_size = fig.get_size_inches()
        fig_aspect = fig_size[1]/fig_size[0]
        sub_ax_height = 0.02
        sub_ax_width = (l_3d + 1.0) * sub_ax_height * 0.5 * (1.0/fig_aspect)
        sub_ax_anchor_x_fig_coords, sub_ax_anchor_y_fig_coords = \
            ax.transData.transform((l, f))
            #ax.transData.inverted().transform((l, f))
        sub_ax_anchor_x, sub_ax_anchor_y = \
            ax.transAxes.inverted().transform(
                    (sub_ax_anchor_x_fig_coords, sub_ax_anchor_y_fig_coords))
        sub_ax_x_min = sub_ax_anchor_x
        sub_ax_y_min = sub_ax_anchor_y - (0.5 * sub_ax_height)
        if (multiplet_offset_dict is not None) and (multiplet in multiplet_offset_dict):
            
            sub_ax_x_offset, sub_ax_y_offset = multiplet_offset_dict[multiplet]
            sub_ax_x_min = sub_ax_x_min + sub_ax_x_offset
            sub_ax_y_min = sub_ax_y_min + sub_ax_y_offset

        sub_ax = ax.inset_axes([sub_ax_x_min, sub_ax_y_min, sub_ax_width, sub_ax_height])

        # Prepare pcolor array.
        pcolor_array = np.zeros((4, (2 * (l_3d + 1)))) + np.nan
        for i in range(num_pts):

            abs_m = np.abs(m_3d[i])

            if m_3d[i] == 0:

                row_0 = 1
                row_1 = 3

                #pcolor_array[1 : 3, 0 : 2] = 1.0

            elif m_3d[i] < 0:

                row_0 = 0
                row_1 = 2
                
                #pcolor_array[0 : 2, 2*abs_m : 2*(abs_m + 1)] = 0.0

            elif m_3d[i] > 0:

                row_0 = 2
                row_1 = 4
                
                #pcolor_array[2 : 4, 2*m_3d[i] : 2*(m_3d[i] + 1)] = 2.0

            pcolor_array[row_0 : row_1, 2*abs_m : 2*(abs_m + 1)] = \
                    coeff_amp_normalised[i]

        h_pcm = sub_ax.pcolormesh(pcolor_array, cmap = c_map, norm = c_norm)

        # Add grid lines.
        make_sub_ax_grid(sub_ax, l_3d)    

        # Tidy up sub axis.
        sub_ax.set_xlim([0.0, 2.0*(l_3d + 1.0)])
        sub_ax.set_ylim([0.0, 4.0])
        #
        sub_ax.xaxis.set_visible(False)
        sub_ax.yaxis.set_visible(False)
        #
        for axis in ['right', 'left', 'top', 'bottom']:

            sub_ax.spines[axis].set_visible(False)

        sub_ax.patch.set_alpha(0.0)

        # Prepare for next iteration.
        i0 = i1

    # Create color bar.
    if colorbar:

        c_ax = ax.inset_axes([0.7, 0.075, 0.25, 0.01])
        plt.colorbar(mappable = h_pcm, cax = c_ax, ax = ax,
                        orientation = 'horizontal', label = 'Coefficient')

    return

def get_max_amp(mode_info_3d):

    # Find maximum excitation amplitude.
    max_amp = 0.0
    for mode_type in mode_info_3d.keys():
        
        if mode_info_3d[mode_type] is None:

            continue

        for multiplet in mode_info_3d[mode_type].keys():

            amp_list = np.sqrt( mode_info_3d[mode_type][multiplet]['coeff_real']**2.0 + \
                                mode_info_3d[mode_type][multiplet]['coeff_imag']**2.0)
            max_amp_i = np.max(amp_list)
            #print('{:} {:>15.3f}'.format(multiplet, 1.0E9*max_amp_i))
            if max_amp_i > max_amp:

                max_amp = max_amp_i

    return max_amp

def make_sub_ax_grid(sub_ax, l):
    
    if l > 0:

        # Vertical lines.
        seg_x = np.zeros(((l + 1), 2))
        seg_y = np.zeros(((l + 1), 2))
        segs  = np.zeros(((l + 1), 2, 2))
        seg_x[:, 0] = seg_x[:, 0] + (2.0 * np.array(range(1, l + 2)))
        seg_x[:, 1] = seg_x[:, 0]
        seg_y[:, 0] = 0.0
        seg_y[:, 1] = 4.0
        segs[:, :, 0] = seg_x 
        segs[:, :, 1] = seg_y
        segs_vert = segs
        # Horizontal lines.
        seg_x = np.zeros((3, 2))
        seg_y = np.zeros((3, 2))
        segs  = np.zeros((3, 2, 2))
        seg_x[:, 0] = 2.0 
        seg_x[:, 1] = 2.0*(l + 1.0)
        seg_y[:, 0] = [0.0, 2.0, 4.0]
        seg_y[:, 1] = seg_y[:, 0]
        segs[:, :, 0] = seg_x 
        segs[:, :, 1] = seg_y
        segs_horz = segs
        # Box at start.
        seg_x = np.zeros((3, 2))
        set_y = np.zeros((3, 2))
        segs  = np.zeros((3, 2, 2))
        seg_x[0, :] = [0.0, 2.0] # Top.
        seg_y[0, :] = [3.0, 3.0] 
        seg_x[1, :] = [0.0, 2.0] # Bottom.
        seg_y[1, :] = [1.0, 1.0]
        seg_x[2, :] = [0.0, 0.0] # Left.
        seg_y[2, :] = [1.0, 3.0]
        segs[:, :, 0] = seg_x
        segs[:, :, 1] = seg_y
        segs_box = segs
        #
        segs = np.concatenate([segs_vert, segs_horz, segs_box])

    else:

        # Box at start.
        seg_x = np.zeros((4, 2))
        seg_y = np.zeros((4, 2))
        segs  = np.zeros((4, 2, 2))
        seg_x[0, :] = [0.0, 2.0] # Top.
        seg_y[0, :] = [3.0, 3.0] 
        seg_x[1, :] = [0.0, 2.0] # Bottom.
        seg_y[1, :] = [1.0, 1.0]
        seg_x[2, :] = [0.0, 0.0] # Left.
        seg_y[2, :] = [1.0, 3.0]
        seg_x[3, :] = [2.0, 2.0] # Right.
        seg_y[3, :] = [1.0, 3.0]
        segs[:, :, 0] = seg_x
        segs[:, :, 1] = seg_y
    #
    line_segments = LineCollection(segs, colors = 'k', clip_on = False)
    sub_ax.add_collection(line_segments)

    return

def plot_projection_wrapper(dir_projections, dir_plots, mode_info_1d, f_3d, i_mode, show = True):

    # Get projection coefficients of 3-D modes.
    file_projection = 'mode_{:>05d}.txt'.format(i_mode)
    path_projection = os.path.join(dir_projections, file_projection)
    mode_info_3d = load_3d_mode_info(path_projection)

    # Get maximum amplitude.
    max_amp = get_max_amp(mode_info_3d)

    # Define axis frequency limits.
    f_min_3d = np.min(f_3d)
    f_max_3d = np.max(f_3d)
    f_range = f_max_3d - f_min_3d
    #y_buff = 0.05
    #f_lim_min = f_min_3d + y_buff*f_range
    #f_lim_max = f_max_3d + y_buff*f_range
    f_buff = 0.25
    f_lim_min = f_min_3d - f_buff
    f_lim_max = f_max_3d + f_buff
    f_lims = [f_lim_min, f_lim_max]

    # Define axis angular order limits.
    l_lim_max = get_max_l(mode_info_1d, f_lims)
    #x_buff = 0.05
    x_buff = 0.6
    l_lims = [-0.5, (1.0 + x_buff)*l_lim_max]

    # Create mapping between 1-D mode type and axis.
    #mode_type_to_axis = {'R' : 0, 'S' : 0, 'T1' : 1, 'T0' : 2}
    mode_type_to_axis = {'R' : 0, 'S' : 0, 'T1' : 1, 'T0' : 1}

    # Create color key for different mode types.
    default_color = 'green'
    backup_color = 'crimson'
    mode_color_dict = { 'R' : default_color, 'S' : default_color,
            'T1' : default_color, 'T0': backup_color}

    # Create figure.
    fig, ax_arr = plt.subplots(1, 2, figsize = (14.0, 11.0), sharex = True,
                        sharey = True, constrained_layout = True)

    # Set axis limits.
    ax = ax_arr[0]
    ax.set_xlim(l_lims)
    ax.set_ylim(f_lims)

    # Draw frequency of specific mode.
    for ax in ax_arr:

        ax.axhline(f_3d[i_mode - 1])

    # Plot underlying mode diagram.
    for mode_type in mode_info_1d.keys():

        # Find axis.
        ax = ax_arr[mode_type_to_axis[mode_type]]

        # Get color.
        color = mode_color_dict[mode_type]
        
        # Plot. 
        plot_dispersion(ax, mode_type, mode_info_1d[mode_type], color = color)

    # Plot excitation.
    for mode_type in mode_info_1d.keys():

        # Find axis.
        ax = ax_arr[mode_type_to_axis[mode_type]]

        if mode_type == 'S':

            colorbar = True

        else:

            colorbar = False

        # Plot. 
        plot_excitation_wrapper(ax, mode_type, mode_info_1d[mode_type], mode_info_3d[mode_type], max_amp,
            custom_offsets = True, colorbar = colorbar)
    
    # Label axes.
    font_size_label = 13
    ax = ax_arr[0]
    ax.set_ylabel("Frequency (mHz)", fontsize = font_size_label)
    for ax in ax_arr:

        ax.set_xlabel("Angular order, $\ell$", fontsize = font_size_label)

    ax_titles = ['Spheroidal', 'Toroidal']
    for ax, ax_title in zip(ax_arr, ax_titles):

        ax.set_title(ax_title, fontsize = font_size_label)

    # Save.
    name_out = 'projection_{:>05d}.png'.format(i_mode)
    path_out = os.path.join(dir_plots, name_out)
    print('Saving to {:}'.format(path_out))
    plt.savefig(path_out, dpi = 300)

    # Show.
    if show:

        plt.show()

    # Close.
    plt.close()

    return

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref", help = 'Path to input file used to calculate 1-D reference model.') 
    parser.add_argument("i_mode", help = 'Integer ID of 3-D mode to plot, or \'all\'.') 
    parser.add_argument("--link_diagram", action = 'store_true', help = "Include this flag to plot a diagram of linked singlets instead of coefficient highlights.")
    args = parser.parse_args()
    path_1d_input = args.path_ref
    link_diagram = args.link_diagram
    i_mode = args.i_mode

    # Load 1-D mode information.
    mode_info_1d, model_1d, run_info_1d = load_1d_mode_info(path_1d_input)

    # Read 3D input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii  = read_input_NMPostProcess()

    # Get output directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_projections = os.path.join(dir_processed, 'projections')
    dir_plots = os.path.join(dir_processed, 'plots')

    # Get a list of 3-D modes.
    i_mode_list = get_list_of_modes_from_coeff_files(dir_NM, option) 
    num_modes = len(i_mode_list)

    # Get frequencies of 3-D modes.
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f_3d = read_eigenvalues(file_eigval_list)
    
    if i_mode == 'all':
        
        show = False

    else:

        i_mode_list = [int(i_mode)]
        show = True

    for i_mode in i_mode_list:

        if link_diagram:

            plot_link_diagram(dir_projections, dir_plots, mode_info_1d,
                    f_3d, i_mode, show = show)

        else:

            plot_projection_wrapper(dir_projections, dir_plots, mode_info_1d,
                    f_3d, i_mode, show = show)

    return

if __name__ == '__main__':

    main()
