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
from NMPostProcess.project_into_mode_basis import (get_max_amp, load_1d_mode_info,
                        load_3d_mode_info, get_singlet_amplitude)

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

def plot_link_diagram_wrapper(dir_projections, dir_plots, mode_info_1d, f_3d, i_mode_list, show = True, path_link_list = None):

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
    x_buff = 0.05
    #x_buff = 0.6
    l_lims = [-0.5, (1.0 + x_buff)*l_lim_max]

    # Create mapping between 1-D mode type and axis.
    #mode_type_to_axis = {'R' : 0, 'S' : 0, 'T1' : 1, 'T0' : 2}
    mode_type_to_axis = {'R' : 0, 'S' : 0, 'T1' : 1, 'T0' : 1}

    # Create color key for different mode types.
    color_R = 'slateblue'
    color_S = color_R
    color_T0 = 'orangered'
    color_T1 = 'forestgreen'
    mode_color_dict = { 'R' : color_R, 'S' : color_S,
            'T1' : color_T1, 'T0': color_T0}
    
    # Create figure.
    fig = plt.figure(figsize = (14.0, 11.0), constrained_layout = True)
    ax  = plt.gca()

    # Set axis limits.
    ax.set_xlim(l_lims)
    ax.set_ylim(f_lims)

    # Draw frequency of specific mode.
    if len(i_mode_list) == 1:

        ax.axhline(f_3d[i_mode_list[0] - 1], color = 'k', linestyle = ':')

    # Plot underlying mode diagram.
    for mode_type in mode_info_1d.keys():

        # Get color.
        color = mode_color_dict[mode_type]
        
        # Plot. 
        plot_dispersion(ax, mode_type, mode_info_1d[mode_type], color = color)
    
    if len(i_mode_list) == 1:

        alpha = 1.0

    else:
        
        alpha = 0.2
        #alpha = 1.0 / len(i_mode_list)
        #min_alpha = (1.0/256.0) # PyPlot can't handle small values of alpha
        #                        # for certain backends.
        min_alpha = 0.01 # Small alpha values are also hard to see.
        if alpha < min_alpha:
            alpha = min_alpha

    # Load link list.
    if path_link_list is not None:

        link_list = load_link_list(path_link_list)

    for i_mode in i_mode_list:

        # Get projection coefficients of 3-D modes.
        file_projection = 'mode_{:>05d}.txt'.format(i_mode)
        path_projection = os.path.join(dir_projections, file_projection)
        mode_info_3d = load_3d_mode_info(path_projection)

        # Get maximum amplitude.
        max_amp = get_max_amp(mode_info_3d)

        # Get the amplitude in each singlet.
        singlet_amplitude = get_singlet_amplitude(mode_info_3d, max_amp)

        if path_link_list is not None:
            
            mode_is_linked = check_if_mode_is_linked(singlet_amplitude, link_list)
            if mode_is_linked:

                plot_link_diagram(ax, mode_info_1d, singlet_amplitude, alpha = alpha)

        else:

            # Plot links.
            plot_link_diagram(ax, mode_info_1d, singlet_amplitude, alpha = alpha)

    # Label axes.
    font_size_label = 13
    ax.set_ylabel("Frequency (mHz)", fontsize = font_size_label)
    ax.set_xlabel("Angular order, $\ell$", fontsize = font_size_label)

    # Save.
    if len(i_mode_list) == 1:

        name_out = 'link_diagram_{:>05d}.png'.format(i_mode)

    else:

        name_out = 'link_diagram_{:>05d}_to_{:>05d}.png'.format(i_mode_list[0], i_mode_list[-1])
    path_out = os.path.join(dir_plots, name_out)
    print('Saving to {:}'.format(path_out))
    plt.savefig(path_out, dpi = 300)

    # Show.
    if show:

        plt.show()

    # Close.
    plt.close()

    return

def plot_link_diagram(ax, mode_info_1d, singlet_amplitude, alpha = 1.0):

    # Count the number of singlets.
    num_singlets = 0
    for type_nl_key in singlet_amplitude.keys():
        
        num_singlets = num_singlets + 1

    # Prepare plotting arrays.
    mode_type_list = []
    n_singlet   = np.zeros(num_singlets, dtype = np.int)
    l_singlet   = np.zeros(num_singlets, dtype = np.int)
    f_singlet   = np.zeros(num_singlets)
    amp_singlet = np.zeros(num_singlets)

    # Fill plotting arrays.
    for i, type_nl_key in enumerate(singlet_amplitude.keys()):

        mode_type, n_str, l_str = type_nl_key.split('_')
        n_singlet[i] = int(n_str)
        l_singlet[i] = int(l_str)

        k = np.where((mode_info_1d[mode_type]['n'] == n_singlet[i]) &
                     (mode_info_1d[mode_type]['l'] == l_singlet[i]))[0]
        
        f_singlet[i]   = mode_info_1d[mode_type]['f'][k]
        amp_singlet[i] = singlet_amplitude[type_nl_key]

        mode_type_list.append(mode_type)

        #print(type_nl_key, n_singlet[i], l_singlet[i], f_singlet[i], amp_singlet[i])

    # Filter small amplitudes.
    #amp_thresh = 0.1
    amp_thresh = 0.3
    i_thresh = np.where(amp_singlet > amp_thresh)[0]
    n_singlet   = n_singlet[i_thresh]
    l_singlet   = l_singlet[i_thresh]
    f_singlet   = f_singlet[i_thresh]
    amp_singlet = amp_singlet[i_thresh]
    mode_type_list = [mode_type_list[i] for i in i_thresh]

    # Count filtered list.
    num_singlet = len(l_singlet)

    # Sort by frequency, then by l-value.
    i_sort = np.argsort(f_singlet)
    n_singlet   = n_singlet[i_sort]
    l_singlet   = l_singlet[i_sort]
    f_singlet   = f_singlet[i_sort]
    amp_singlet = amp_singlet[i_sort]
    mode_type_list = [mode_type_list[i] for i in i_sort]
    #
    i_sort = np.argsort(l_singlet)
    n_singlet   = n_singlet[i_sort]
    l_singlet   = l_singlet[i_sort]
    f_singlet   = f_singlet[i_sort]
    amp_singlet = amp_singlet[i_sort]
    mode_type_list = [mode_type_list[i] for i in i_sort]

    for i in range(num_singlet):
        print('{:>2} {:>3d} {:>3d}'.format(mode_type_list[i], n_singlet[i],
                l_singlet[i]), end = ' ')
    print('\n', end = '')
    
    # Get size of markers.
    s_max = 50.0
    max_amp_singlet = np.max(amp_singlet)
    s = s_max * (amp_singlet/max_amp_singlet)
    
    # Scatter plot.
    ax.scatter(l_singlet, f_singlet, s = s, c = 'k', zorder = 20, alpha = alpha)

    # Line plot.
    if num_singlet > 1:

        ax.plot(l_singlet, f_singlet, c = 'k', alpha = alpha)

    return

def load_link_list(path_link_list):
    
    link_list = []
    with open(path_link_list, 'r') as in_id:

        for line in in_id.readlines():
            
            mode_type, n_str, l_str = line.split()
            n = int(n_str)
            l = int(l_str)
            mode_type_nl_tag = '{:}_{:>05d}_{:05d}'.format(mode_type, n, l)

            link_list.append(mode_type_nl_tag)

    return link_list

def check_if_mode_is_linked(singlet_amplitude, link_list, thresh = 0.1):

    amp_of_link_list = []
    for mode in link_list:

        if mode in singlet_amplitude.keys():

            amp_of_link_list.append(singlet_amplitude[mode])

        else:

            amp_of_link_list.append(0.0)
    
    above_thresh = [(amp > thresh) for amp in amp_of_link_list]
    if any(above_thresh):

        return True

    return False

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref", help = 'Path to input file used to calculate 1-D reference model.') 
    parser.add_argument("i_mode", help = 'Integer ID of 3-D mode to plot, or \'all\'.') 
    parser.add_argument("--link_diagram", action = 'store_true', help = "Include this flag to plot a diagram of linked singlets instead of coefficient highlights.")
    parser.add_argument("--link_overlay", action = 'store_true', help = "Include this flag along with --link_diagram to plot all links on a single diagram.")
    parser.add_argument("--path_link_list", help = "Include this flag along with --link_diagram and --link_overlay to plot all links of the modes specified in path_link_list on a single diagram.")
    args = parser.parse_args()
    path_1d_input = args.path_ref
    link_diagram = args.link_diagram
    link_overlay = args.link_overlay
    path_link_list = args.path_link_list
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

    elif ',' in i_mode:

        i_mode_0_str, i_mode_1_str = i_mode.split(',')
        i_mode_0 = int(i_mode_0_str)
        i_mode_1 = int(i_mode_1_str)
        i_mode_list = list(range(i_mode_0, i_mode_1 + 1))
        show = False

    else:

        i_mode_list = [int(i_mode)]
        show = True

    if link_diagram:

        if link_overlay:

            plot_link_diagram_wrapper(dir_projections, dir_plots, mode_info_1d,
                    f_3d, i_mode_list, show = show, path_link_list = path_link_list)

        else:

            for i_mode in i_mode_list:

                plot_link_diagram_wrapper(dir_projections, dir_plots, mode_info_1d,
                        f_3d, [i_mode], show = show)

    else:
        
        for i_mode in i_mode_list:

            plot_projection_wrapper(dir_projections, dir_plots, mode_info_1d,
                        f_3d, i_mode, show = show)

    return

if __name__ == '__main__':

    main()
