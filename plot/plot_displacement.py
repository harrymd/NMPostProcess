# Core modules.
import os

# Third-party modules.
import  cartopy.crs             as      ccrs
from    cartopy.mpl.geoaxes     import  GeoAxes
from    cartopy.util            import  add_cyclic_point
import  matplotlib              as      mpl
import  matplotlib.pyplot       as      plt
from    matplotlib.ticker       import  MaxNLocator, MultipleLocator, IndexLocator
from    mpl_toolkits.axes_grid1 import  AxesGrid
import  numpy                   as      np
import  shtns

# Local modules.
from common         import convert_complex_sh_to_real, get_list_of_modes_from_output_files, load_vsh_coefficients, make_l_and_m_lists, mkdir_if_not_exist
from process        import project_from_spherical_harmonics 

def region_int_to_title(region, radius, shell_name_path = None):

    if shell_name_path is not None:

        try:

            with open(shell_name_path, 'r') as in_id:

                shell_names = [x.strip() for x in in_id.readlines()]

        except FileNotFoundError:

            print('Could not find shell name file {:}'.format(shell_name_path))
            shell_names = None

    else:

        shell_names = None 
    
    i_shell = region//3

    if shell_names is None:

        shell_str = 'shell {:>1d}'.format(i_shell)

    else:

        shell_str = shell_names[i_shell]

    if region % 3 == 0:

        if region == 0:

            title = 'Outer surface'

        else:

            title = 'Outer surface of {:}'.format(shell_str)

    elif region % 3 == 1:
        
        title = 'Interior of {:}, at radius {:.1f}'.format(shell_str, radius)

    elif region % 3 == 2:

        title = 'Inner surface of {:}'.format(shell_str)

    return title

def save_figure(path_fig, fmt):

    if fmt == 'pdf':

        plt.savefig(path_fig)

    elif fmt == 'png':

        plt.savefig(path_fig, dpi = 300)

    else:

        raise NotImplementedError('Can only save to format pdf or png, not {:}'.format(fmt))

    return

# Plot displacement patterns. -------------------------------------------------
def plot_sh_disp_all_modes(dir_NM, n_lat_grid, fmt = 'pdf'):

    mode_list = get_list_of_modes_from_output_files(dir_NM)
    
    for i_mode in mode_list:

        print('\nPlotting displacement for mode {:>5d}.'.format(i_mode))

        plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, show = False, fmt = fmt)

    return

def plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, show = True, fmt = 'pdf'):

    # Reconstruct the coordinate grid.
    n_lon_grid = (2*n_lat_grid) - 1
    lon_grid = np.linspace(0.0, 2.0*np.pi, n_lon_grid + 1, endpoint = True)[:-1]
    lat_grid = np.linspace(-np.pi/2.0, np.pi/2.0, n_lat_grid, endpoint=True)

    # Load the VSH coefficients for the specified mode.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    Ulm, Vlm, Wlm, scale, r_max, region_max = load_vsh_coefficients(dir_NM, i_mode)
    title = region_int_to_title(region_max, r_max, shell_name_path = os.path.join(dir_processed, 'shell_names.txt'))

    # Infer the maximum l-value used.
    n_coeffs = len(Ulm)
    l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

    # Re-construct the shtns calculator.
    m_max = l_max
    sh_calculator   = shtns.sht(l_max, m_max)
    grid_type       =       shtns.sht_reg_fast          \
                        |   shtns.SHT_SOUTH_POLE_FIRST  \
                        |   shtns.SHT_PHI_CONTIGUOUS
    sh_calculator.set_grid(n_lat_grid, n_lon_grid, flags = grid_type)

    # Expand the VSH into spatial components.
    U_r, V_e, V_n, W_e, W_n = project_from_spherical_harmonics(sh_calculator, Ulm, Vlm, Wlm)

    #fig, ax_arr = plt.subplots(1, 3, figsize = (10.0, 3.5))

    ## Define the colour map.
    #c_interval          = 0.1
    #c_min               = -1.0
    #c_max               =  1.0
    #c_levels      = np.linspace(c_min, c_max, num = 21)
    ## 
    #c_norm    = mpl.colors.Normalize(
    #                    vmin = c_min,
    #                    vmax = c_max)
    ##            
    #c_map     = plt.get_cmap('seismic')
    #c_map.set_over('green')
    #c_map.set_under('green')
    #
    #ax = ax_arr[0]
    #conts = ax.contourf(
    #    U_r,
    #    c_levels,
    #    norm        = c_norm,
    #    cmap        = c_map)

    #V = np.sqrt((V_e**2.0) + (V_n**2.0))
    #ax = ax_arr[1]
    #conts = ax.contourf(
    #    V,
    #    c_levels,
    #    norm        = c_norm,
    #    cmap        = c_map)
    ##ax_arr[1].contourf(V)

    #W = np.sqrt((W_e**2.0) + (W_n**2.0))
    #ax = ax_arr[2]
    #conts = ax.contourf(
    #    W,
    #    c_levels,
    #    norm        = c_norm,
    #    cmap        = c_map)
    #ax_arr[1].contourf(W)

    #plt.show()

    #import sys
    #sys.exit()
    
    # Plot.
    # The contourf() function can suffer an error when the input is almost flat (e.g. plotting the toroidal component of a spheroidal mode). To avoid this error, we reduce the number of contour levels.
    n_c_levels_default = 11 
    n_c_levels_fallback = 6
    try:

        plot_sh_disp_3_comp(lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default)

    except AttributeError:

        print("Encountered Cartopy attribute error, trying with fewer contour levels.")
        plt.close()
        plot_sh_disp_3_comp(lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_fallback)
        #raise

    # Save the plot.
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)
    name_fig = 'displacement_quick_{:>05d}.{:}'.format(i_mode, fmt)
    path_fig = os.path.join(dir_plot, name_fig)
    print('Saving figure to {:}'.format(path_fig))
    save_figure(path_fig, fmt)
    plt.savefig(path_fig)

    # Show the plot.
    if show:

        plt.show()

    # Close the plot.
    plt.close()

    return

def plot_sh_disp(lon_grid, lat_grid, v_r = None, v_n = None, v_e = None, ax = None, ax_cbar = None, n_c_levels = 21):
    
    # Check that the right combination of components was provided.
    err_message_comps = 'Must provide either the radial component, or both the north and east components (for a tangential plot)' 
    if v_r is not None:

        components = 'radial'
        assert ((v_e is None) and (v_n is None)), err_message_comps

    else:

        components = 'tangential'
        assert ((v_e is not None) and (v_n is not None)), err_message_comps
    
    # Define the scalar variable for the contour map.
    if components == 'radial':

        v_scalar = v_r

    elif components == 'tangential':

        v_scalar = np.sqrt((v_e**2.0) + (v_n**2.0))

    # Define the colour map.
    c_interval          = 0.1
    c_min               = -1.0
    c_max               =  1.0
    c_levels      = np.linspace(c_min, c_max, num = n_c_levels)
    # 
    c_norm    = mpl.colors.Normalize(
                        vmin = c_min,
                        vmax = c_max)
    #            
    c_map     = plt.get_cmap('seismic')
    c_map.set_over('green')
    c_map.set_under('green')
    
    # Create the axes if necessary.
    if ax is None:

        ax  = plt.axes(projection = ccrs.Mollweide())
        ax.set_global()

    # Add an extra point to avoid a seam at the prime meridian.
    v_scalar, lon_grid = add_cyclic_point(v_scalar, lon_grid)

    # Plot the contours.
    conts = ax.contourf(
        np.rad2deg(lon_grid),
        np.rad2deg(lat_grid),
        v_scalar,
        c_levels,
        transform   = ccrs.PlateCarree(),
        norm        = c_norm,
        cmap        = c_map)
    
    # Plot the arrows, if it is a tangential map.
    if components == 'tangential': 
        
        # slice_ controls the down-sampling of the data.
        slice_ = int(np.ceil(lon_grid.shape[0]/20))
        ax.quiver(
            np.rad2deg(lon_grid[::slice_]),
            np.rad2deg(lat_grid[::slice_]),
            v_e[::slice_, ::slice_],
            v_n[::slice_, ::slice_],
            transform   = ccrs.PlateCarree(),
            scale       = 1.0E1,
            pivot       = 'middle',
            color       = 'mediumblue')

    # Add the colour bar.
    if ax_cbar is not None:

        c_bar = plt.colorbar(
                    conts,
                    cax         = ax_cbar,
                    orientation = 'horizontal')
        #c_bar.set_label('Magnitude of displacement', fontsize = 12)
        c_bar.set_label('Displacement', fontsize = 12)
    
    return

def plot_sh_disp_3_comp(lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, ax_arr = None, show = True, title = None, n_c_levels = 21): 
    
    # Create axis array.
    # It must have three axes and the third axis must have the cax (colorbar axis) attribute.
    if ax_arr is None:
        
        # Create the axes.
        projection = ccrs.Mollweide()
        fig = plt.figure(figsize = (3.5, 6.0))
        axes_class = (GeoAxes, dict(map_projection = projection))
        ax_arr = AxesGrid(fig, 111,
                axes_class = axes_class,
                nrows_ncols=(3, 1),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='bottom',
                label_mode = '',
                cbar_pad=0.1,
                )

    # Unpack the axis array.
    ax_U, ax_V, ax_W = ax_arr
    
    # Plot the displacement of each component.
    # The contourf() function can suffer an error when the input is almost flat (e.g. plotting the toroidal component of a spheroidal mode). To avoid this error, we reduce the number of contour levels.
    #n_c_levels = 21
    #n_c_levels_backup = 11
    #try:
    #    plot_sh_disp(lon_grid, lat_grid, v_r = U_r,             ax = ax_U, n_c_levels = n_c_levels)
    #except AttributeError: 
    #    plot_sh_disp(lon_grid, lat_grid, v_r = U_r,             ax = ax_U, n_c_levels = n_c_levels_backup)

    #try:
    #    plot_sh_disp(lon_grid, lat_grid, v_e = V_e, v_n = V_n,  ax = ax_V, n_c_levels = n_c_levels)
    #except AttributeError: 
    #    plot_sh_disp(lon_grid, lat_grid, v_e = V_e, v_n = V_n,  ax = ax_V, n_c_levels = n_c_levels_backup)

    #try:
    #    print('Aaaa')
    #    plot_sh_disp(lon_grid, lat_grid, v_e = W_e, v_n = W_n,  ax = ax_W, n_c_levels = n_c_levels, ax_cbar = ax_W.cax)
    #except AttributeError: 
    #    print('BBB')
    #    plot_sh_disp(lon_grid, lat_grid, v_e = W_e, v_n = W_n, ax = ax_W, n_c_levels = n_c_levels_backup, ax_cbar = ax_W.cax)

    # This is the simpler version.
    #n_c_levels = 11
    plot_sh_disp(lon_grid, lat_grid, v_r = U_r,             ax = ax_U, n_c_levels = n_c_levels)
    plot_sh_disp(lon_grid, lat_grid, v_e = V_e, v_n = V_n,  ax = ax_V, n_c_levels = n_c_levels)
    plot_sh_disp(lon_grid, lat_grid, v_e = W_e, v_n = W_n,  ax = ax_W, n_c_levels = n_c_levels, ax_cbar = ax_W.cax)

    if title is not None:

        plt.suptitle(title)
    
    # Tidy around axes.
    plt.tight_layout()
    
    # Show (if requested).
    if show:
        
        plt.show()

    return

# Plot spectral data. ---------------------------------------------------------
def plot_sh_real_coeffs_3_comp_all_modes(dir_NM, fmt = 'pdf'):

    mode_list = get_list_of_modes_from_output_files(dir_NM)
    
    for i_mode in mode_list:

        print('\nPlotting spectrum for mode {:>5d}.'.format(i_mode))

        plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, show = False, fmt = fmt)

    return

def plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, show = True, fmt = 'pdf'):

    # Load VSH coefficients.
    dir_processed = os.path.join(dir_NM, 'processed')
    Ulm, Vlm, Wlm, scale, r_max, region_max = load_vsh_coefficients(dir_NM, i_mode)
    title = region_int_to_title(region_max, r_max, shell_name_path = os.path.join(dir_processed, 'shell_names.txt'))
    # Infer the maximum l-value used.
    n_coeffs = len(Ulm)
    l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1
    
    # Convert from complex to real form.
    ulm, l, m = convert_complex_sh_to_real(Ulm, l_max)
    vlm, _, _ = convert_complex_sh_to_real(Vlm, l_max)
    wlm, _, _ = convert_complex_sh_to_real(Wlm, l_max)

    ## Generate the l and m lists.
    #l, m = make_l_and_m_lists(l_max)

    # Plot the coefficients.
    plot_sh_real_coeffs_3_comp(
            l, m, ulm, vlm, wlm,
            title_str   = title,
            fig         = None,
            ax_arr      = None,
            show        = False,
            flip        = True,
            c_max       = None,
            abs_plot    = False)

    # Save the plot.
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)
    name_fig = 'spectrum_quick_{:>05d}.{:}'.format(i_mode, fmt)
    path_fig = os.path.join(dir_plot, name_fig)
    print('Saving figure to {:}'.format(path_fig))

    save_figure(path_fig, fmt)
    
    # Show the figure.
    if show:

        plt.show()

    # Close the figure.
    plt.close()

    return 

def plot_sh_real_coeffs(l_list, m_list, coeffs, c_lims = None, ax = None, show = True, add_cbar = True, label = None, l_lims_plot = None, flip = False, stack_vertical = False, in_ticks = False, abs_plot = True, x_label = 'Angular order, $\ell$', y_label = 'Azimuthal order, $m$'):
    
    # Put the list of coefficients onto a grid, for plotting.
    # k Index in list.
    # i x-index in grid.
    # j y-index in grid.
    # l l-value in list.
    # m m-value in list.
    l_max = np.max(l_list)
    m_max = np.max(m_list)
    #
    coeff_grid          = np.zeros((l_max + 1, 2*m_max + 1))
    coeff_grid[:, :]    = np.nan
    # 
    k = 0
    for l in range(l_max + 1):
        
        i = l
        
        for m in range(-l, (l + 1)):
            
            # Using - m instead of + m agrees with SHTools, but I don't see
            # why.
            j = l_max + m
            coeff_grid[i, j] = coeffs[k]

            k = k + 1
    
    # Calculate absolute values.
    abs_coeff_grid = np.abs(coeff_grid)
    
    # Define the corners of the cells for a pcolor plot.
    l_corners = np.array(range(l_max + 2)) - 0.5
    m_corners = np.array(range(-(m_max), (m_max + 2))) - 0.5

    # Create axes if necessary.
    if ax is None:
        
        fig = plt.figure()
        ax  = plt.gca()
    
    # Create the color map.
    if abs_plot: 
        
        c_map     = plt.get_cmap('plasma')

    else:

        c_map   = plt.get_cmap('seismic')

    c_map.set_bad('grey')

    #
    if c_lims is None:
        
        if abs_plot:
            
            c_lims = [0.0, np.nanmax(abs_coeff_grid)]

        else:

            c_max = np.nanmax(abs_coeff_grid)
            c_lims = [-c_max, c_max]
    # 
    c_norm          = mpl.colors.Normalize(
                         vmin = c_lims[0],
                         vmax = c_lims[1])
    
    if abs_plot:

        array = abs_coeff_grid

    else:

        array = coeff_grid

    # Plot the coefficients.
    # The choice of axes can be flipped.
    if flip:

        image = ax.pcolormesh(m_corners, l_corners, array, 
                                norm = c_norm,
                                cmap = c_map)

    else:

        image = ax.pcolormesh(l_corners, m_corners, array.T,
                                norm = c_norm,
                                cmap = c_map)
    
    # Aspect 1.0 is preferable, but can be hard for arranging subplots.
    #ax.set_aspect(1.0)

    # Apply the axis limits.
    if l_lims_plot is None:

        l_lims_plot = [0.0, l_max]
        
    if flip:

        ax.set_ylim([-0.5, l_max + 0.5])
        ax.set_xlim([-(m_max + 0.5), m_max + 0.5])

    else:

        ax.set_xlim([-0.5, l_max + 0.5])
        ax.set_ylim([-(m_max + 0.5), m_max + 0.5])

    # Create the axis labels.
    font_size_label = 12
    if flip:

        x_label_temp = x_label
        x_label = y_label
        y_label = x_label_temp

    if y_label is not None:

        ax.set_ylabel(y_label, fontsize = font_size_label)

    if x_label is not None:

        ax.set_xlabel(x_label, fontsize = font_size_label)

    # Add the color bar.
    if add_cbar:
        
        c_bar = plt.colorbar(
                image,
                cax         = ax.cax,
                orientation = 'horizontal',)

        if abs_plot:

            c_bar_label = 'Magnitude of coefficients'

        else:
            
            c_bar_label = 'Coefficients'

        c_bar.set_label(c_bar_label, fontsize = font_size_label)
    
    # Force integer ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax.yaxis.set_major_locator(MaxNLocator(integer = True))

    # Create a label on the plot, e.g. |Ulm|.
    if label is not None:
        
        label = '|{}$_{{lm}}|$'.format(label)
        ax.text(0.1, 0.9, label, transform = ax.transAxes)
    
    # Change the ticks and their labels to be on the inside of the
    # x-axis (if requested).
    if in_ticks:

        ax.tick_params(axis = "x", direction = "in", pad = -15)

    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(IndexLocator(base = 1.0, offset = 0.5))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(IndexLocator(base = 1.0, offset = 0.5))
    ax.grid(which = 'major', axis = 'both', alpha = 0.3, linewidth = 1)
    ax.grid(which = 'minor', axis = 'both', alpha = 0.1, linewidth = 1)

    if show:
        
        plt.show()
        
    return image

def plot_sh_real_coeffs_3_comp(l, m, ulm, vlm, wlm, title_str = None, fig = None, ax_arr = None, show = True, flip = False, c_max = None, abs_plot = True): 

    # Create the figure if none was specified.
    if fig is None: 

        assert ax_arr is None, 'Arguments fig and ax_arr must both be given or both be None.'

        # Create the axes.
        fig = plt.figure(figsize = (3.5, 6.0))
        ax_arr = AxesGrid(fig, 111,
                nrows_ncols=(3, 1),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='bottom',
                label_mode = '',
                cbar_pad=0.3,
                )
    
    # Determine the limits of the color bar.
    if c_max is None:

        c_max = np.nanmax(np.abs(np.array([ulm, vlm, wlm])))

    if abs_plot:

        c_lims = [0.0, c_max]

    else:
        
        c_lims = [-c_max, c_max]
    
    # Set the l-limit of the plot.
    l_max_plot = 30 
    l_lims_plot = [0, l_max_plot]
    
    # Plot the coefficients.
    # label = 'U'
    plot_sh_real_coeffs(l, m, ulm, ax = ax_arr[0], label = None, flip = flip, abs_plot = abs_plot, add_cbar = False, show = False, c_lims = c_lims, l_lims_plot = l_lims_plot, x_label = None, y_label = None)
    plot_sh_real_coeffs(l, m, vlm, ax = ax_arr[1], label = None, flip = flip, abs_plot = abs_plot, add_cbar = False, show = False, c_lims = c_lims, l_lims_plot = l_lims_plot, x_label = None, y_label = None)
    plot_sh_real_coeffs(l, m, wlm, ax = ax_arr[2], label = None, flip = flip, abs_plot = abs_plot, show = False, c_lims = c_lims, l_lims_plot = l_lims_plot, stack_vertical = True, in_ticks = True)
    
    # Add a title, if specified.
    if title_str is not None:

        plt.suptitle(title_str, fontsize = 14)

    # Show, if requested.
    if show:

        plt.show()

    return

# Main and sentinel. ----------------------------------------------------------
def main():

    # Read the NMPostProcess input file.
    input_file = 'input_NMPostProcess.txt'
    with open(input_file, 'r') as in_id:

        input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    input_args = [x.strip() for x in input_args]
    dir_PM      = input_args[0]
    dir_NM      = input_args[1]
    option      = input_args[2]
    l_max       = int(input_args[3])
    #i_mode_str  = input_args[4]

    # Read the plotting input file.
    plot_input_file = 'input_plotting.txt'
    with open(plot_input_file, 'r') as in_id:

        plot_input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    plot_input_args = [x.strip() for x in plot_input_args]
    plot_type       = plot_input_args[0]
    i_mode_str      = plot_input_args[1]
    fmt             = plot_input_args[2]
    if i_mode_str == 'all':

        if plot_type == 'spatial':

            n_lat_grid = int(plot_input_args[3])
            plot_sh_disp_all_modes(dir_NM, n_lat_grid, fmt = fmt)

        elif plot_type == 'spectral':

            plot_sh_real_coeffs_3_comp_all_modes(dir_NM, fmt = fmt)

        else:

            raise ValueError('Plot type {:} from input file {:} not recognised.'.format(plot_type, input_file))

    else:

        i_mode = int(i_mode_str)
        if plot_type == 'spatial':

            n_lat_grid = int(plot_input_args[3])
            plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, fmt = fmt)
        

        elif plot_type == 'spectral':
            
            plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, fmt = fmt)

        else:

            raise ValueError('Plot type {:} from input file {:} not recognised.'.format(plot_type, input_file))

if __name__ == '__main__':

    main()