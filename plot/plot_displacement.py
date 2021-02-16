'''
NMPostProcess/plot/plot_displacement.py
Plots spatial and spectral representation of vector-spherical-harmonic fields.
'''

# Import modules. -------------------------------------------------------------

# Core modules.
from glob import glob
import os

# Third-party modules.
import  cartopy.crs             as      ccrs
from    cartopy.mpl.geoaxes     import  GeoAxes
from    cartopy.util            import  add_cyclic_point
import  matplotlib              as      mpl
from    matplotlib.animation    import  FuncAnimation
import  matplotlib.pyplot       as      plt
from    matplotlib.ticker       import  MaxNLocator, MultipleLocator, IndexLocator
from    mpl_toolkits.axes_grid1 import  AxesGrid
import  numpy                   as      np
import  shtns

# Local modules.
from common         import convert_complex_sh_to_real, get_list_of_modes_from_coeff_files, load_vsh_coefficients, make_l_and_m_lists, mkdir_if_not_exist, read_eigenvalues, read_input_NMPostProcess, read_input_plotting
from NMPostProcess.plot.plot_common import save_figure
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

# Plot displacement patterns. -------------------------------------------------
def plot_sh_disp_all_modes(dir_NM, n_lat_grid, mode_real_or_complex, fmt = 'pdf', i_radius_str = None):
    '''
    For each mode, plots a vector field on the surface of a sphere in terms of the radial, consoidal and toroidal components.
    This is a wrapper for plot_sh_disp_wrapper().

    Input:

    dir_NM, n_lat_grid, fmt
        See 'Definitions of variables' in NMPostProcess/process.py.

    Output:

    None
    '''
    
    if i_radius_str is None:
        
        option = 'quick'

    else:

        option = 'full'

    # Get a list of mode IDs to plot.
    mode_list = get_list_of_modes_from_coeff_files(dir_NM, option)

    # Plot the modes one by one.
    for i_mode in mode_list:

        print('\nPlotting displacement for mode {:>5d}.'.format(i_mode))

        plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, mode_real_or_complex, show = False, fmt = fmt, i_radius_str = i_radius_str)

    return

def plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, mode_real_or_complex, show = True, fmt = 'pdf', transparent = False, i_radius_str = None, close = True, c_bar_label = 'Default', figsize = None):
    '''
    Plots a vector field on the surface of a sphere in terms of the radial, consoidal and toroidal components.
    This is a wrapper for plot_sh_disp() which first loads the necessary arrays. 

    Input:

    dir_NM, i_mode, n_lat_grid, fmt
        See 'Definitions of variables' in NMPostProcess/process.py.

    Output:

    None
    '''

    # Set transparency.
    transparent = True

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)

    # Determine if the plot is 'quick' mode or 'full' mode.
    if i_radius_str is None:

        option = 'quick'

    else:

        option = 'full'

    if mode_real_or_complex in ['real', 'complex_imag_only', 'complex_real_only']:
        
        if figsize is None:

            figsize = (3.5, 6.0)

        fig = plt.figure(figsize = figsize)
        two_panel = False
    
    elif mode_real_or_complex == 'complex':

        #fig = plt.figure(figsize = (14.0, 6.0))
        
        fig = plt.figure(figsize = (7.0, 6.0))
        two_panel = True

    else:

        raise ValueError

    # Reconstruct the coordinate grid.
    n_lon_grid = (2*n_lat_grid) - 1
    lon_grid = np.linspace(0.0, 2.0*np.pi, n_lon_grid + 1, endpoint = True)[:-1]
    lat_grid = np.linspace(-np.pi/2.0, np.pi/2.0, n_lat_grid, endpoint=True)

    if i_radius_str == 'all':

        _, _, _, _, _, header_info = load_vsh_coefficients(dir_NM, i_mode, 0)
        n_samples = len(header_info['r_sample'])

        i_radius_list = list(range(n_samples))

    elif i_radius_str is None:

        i_radius_list = [None]

    else:

        i_radius_list = [int(i_radius_str)]

    # Identify the eigenvalue list file.
    # Load the frequency of the mode.
    path_eigenvalues = os.path.join(dir_processed, 'eigenvalue_list.txt')
    freq_mHz = read_eigenvalues(path_eigenvalues, i_mode = i_mode)

    first_iteration = True
    for j_radius in i_radius_list:

        # Load the VSH coefficients for the specified mode.
        coeffs, header_info, r_sample, i_sample = load_vsh_coefficients(dir_NM, i_mode, j_radius)

        if mode_real_or_complex == 'real':

            Ulm, Vlm, Wlm = coeffs

        else:

            Ulm_real, Vlm_real, Wlm_real, Ulm_imag, Vlm_imag, Wlm_imag = coeffs

        title_str_mode = 'Mode {:>4d}'.format(i_mode)
        title_str_freq = 'freq. {:>7.4f} mHz'.format(freq_mHz)
        title_str_sample = region_int_to_title(i_sample, r_sample, shell_name_path = os.path.join(dir_processed, 'shell_names.txt'))
        title_str_sample = title_str_sample[0].lower() + title_str_sample[1:]
        if two_panel:

            title = '{:} ({:}), sampled at {:}'.format(title_str_mode, title_str_freq, title_str_sample)

        else:

            title = '{:} ({:}),\nsampled at {:}'.format(title_str_mode, title_str_freq, title_str_sample)

        if first_iteration:

            # Infer the maximum l-value used.
            #n_coeffs = len(Ulm)
            n_coeffs = coeffs.shape[-1]
            l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

            # Re-construct the shtns calculator.
            m_max = l_max
            sh_calculator   = shtns.sht(l_max, m_max)
            grid_type       =       shtns.sht_reg_fast          \
                                |   shtns.SHT_SOUTH_POLE_FIRST  \
                                |   shtns.SHT_PHI_CONTIGUOUS
            sh_calculator.set_grid(n_lat_grid, n_lon_grid, flags = grid_type)

            first_iteration = False

        n_c_levels_default = 20 

        # Expand the VSH into spatial components and plot.
        if mode_real_or_complex == 'real':

            U_r, V_e, V_n, W_e, W_n = project_from_spherical_harmonics(sh_calculator, Ulm, Vlm, Wlm)
            fig, ax_arr, handles, contour_info = plot_sh_disp_3_comp(lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, fig = fig, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default)

            fields = {  'U' : {'r' : U_r},
                        'V' : {'e' : V_e, 'n' : V_n},
                        'W' : {'e' : W_e, 'n' : W_n}}

        else:

            U_real_r, V_real_e, V_real_n, W_real_e, W_real_n = project_from_spherical_harmonics(sh_calculator, Ulm_real, Vlm_real, Wlm_real)
            U_imag_r, V_imag_e, V_imag_n, W_imag_e, W_imag_n = project_from_spherical_harmonics(sh_calculator, Ulm_imag, Vlm_imag, Wlm_imag)

            print(U_real_r)

            fields = {
                'U' :   {   'real' : {  'r' : U_real_r},
                            'imag' : {  'r' : U_imag_r}},
                'V' :   {   'real' : {  'n' : V_real_n,
                                        'e' : V_real_e},
                            'imag' : {  'n' : V_imag_n,
                                        'e' : V_imag_e}},
                'W' :   {   'real' : {  'n' : W_real_n,
                                        'e' : W_real_e},
                            'imag' : {  'n' : W_imag_n,
                                        'e' : W_imag_e}}}

            #U_abs_r = np.sqrt((U_real_r**2.0) + (U_imag_r**2.0))
            #V_abs_e = np.sqrt((V_real_e**2.0) + (V_imag_e**2.0))
            #V_abs_n = np.sqrt((V_real_n**2.0) + (V_imag_n**2.0))
            #W_abs_e = np.sqrt((W_real_e**2.0) + (W_imag_e**2.0))
            #W_abs_n = np.sqrt((W_real_n**2.0) + (W_imag_n**2.0))

            if c_bar_label == 'Default':

                c_bar_label_real = 'Real part'
                c_bar_label_imag = 'Imaginary part'

            else:

                c_bar_label_real = c_bar_label
                c_bar_label_imag = c_bar_label

            if mode_real_or_complex == 'complex':

                fig, ax_arr_real, handles_real, contour_info = plot_sh_disp_3_comp(lon_grid, lat_grid, U_real_r, V_real_e, V_real_n, W_real_e, W_real_n, fig = fig, axes_grid_int = 121, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default, c_bar_label = c_bar_label_real)
                fig, ax_arr_imag, handles_imag, contour_info = plot_sh_disp_3_comp(lon_grid, lat_grid, U_imag_r, V_imag_e, V_imag_n, W_imag_e, W_imag_n, fig = fig, axes_grid_int = 122, ax_arr = None, show = False, title = None, n_c_levels = n_c_levels_default, c_bar_label = c_bar_label_imag)
            #plot_sh_disp_3_comp(lon_grid, lat_grid, U_abs_r, V_abs_e, V_abs_n, W_abs_e, W_abs_n, fig = fig, axes_grid_int = 143, fig, ax_arr = None, show = False, title = None, n_c_levels = n_c_levels_default)

                ax_arr = [ax_arr_real, ax_arr_imag]
                handles = [handles_real, handles_imag]

            elif mode_real_or_complex == 'complex_real_only':

                fig, ax_arr, handles, contour_info = plot_sh_disp_3_comp(lon_grid, lat_grid, U_real_r, V_real_e, V_real_n, W_real_e, W_real_n, fig = fig, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default, c_bar_label = c_bar_label_real) 

            elif mode_real_or_complex == 'complex_imag_only':

                fig, ax_arr, handles, contour_info = plot_sh_disp_3_comp(lon_grid, lat_grid, U_real_r, V_real_e, V_real_n, W_real_e, W_real_n, fig = fig, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default, c_bar_label = c_bar_label_imag)

            else:
                
                raise ValueError

        # Save the plot.
        if fmt is not None:

            real_or_complex_str_dict = {
                'real'              : 'real',
                'complex'              : 'cplx',
                'complex_real_only'     : 'real_only',
                'complex_imag_only'    : 'imag_only'}

            real_or_complex_str = real_or_complex_str_dict[mode_real_or_complex]

            if option == 'quick':

                name_fig = 'displacement_{:}_{:}_{:>05d}.{:}'.format(option, real_or_complex_str, i_mode, fmt)

            else:

                name_fig = 'displacement_{:}_{:}_{:>05d}_{:>03d}.{:}'.format(option, real_or_complex_str, i_mode, j_radius, fmt)

            path_fig = os.path.join(dir_plot, name_fig)
            print('Saving figure to {:}'.format(path_fig))
            save_figure(path_fig, fmt, transparent = transparent)

        # Show the plot.
        if show:

            plt.show()

        # Close the plot.
        if close:

            plt.close()

    return fig, ax_arr, handles, fields, contour_info

def plot_sh_disp(lon_grid, lat_grid, v_r = None, v_n = None, v_e = None, ax = None, ax_cbar = None, n_c_levels = 20, c_lims = [-1.0, 1.0], c_map_name = 'seismic', c_bar_label = 'Displacement'):
    '''
    Plots the displacement pattern of a radial or tangential vector field on surface of a sphere. If the field is purely radial, the plot shows the magnitude and sign of the displacement. If the field is purely tangential, the plot shows the absolute value of the displacement, with arrows showing the direction.

    Note:

    There are known issues with the contourf() function for Cartopy geoaxes. See NMPostProcess/README.md for more information.

    Input:

    lon_grid, lat_grid
        See 'Definitions of variables' in NMPostProcess/process.py.

    v_r, v_n, v_e
        (n_lat_grid, n_lon_grid) The radial, north and east components. Either specify v_r only (for a radial plot) or v_e and v_n only (for a tangential plot).
    ax
        An axis for plotting. If None, will be created.
    ax_cbar
        The axis for the colour bar. If None, no colour bar shown.
    n_c_levels
        The number of colour bar levels.

    Output:

    None
    '''
    
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
    #c_interval          = 0.1
    #c_min               = -1.0
    #c_max               =  1.0
    c_min, c_max = c_lims
    c_levels      = np.linspace(c_min, c_max, num = n_c_levels)
    half_n_c_levels = n_c_levels//2
    c_levels_pos = np.linspace(0.0, c_max, num = (half_n_c_levels + 1))[1:]
    c_levels_neg = -1.0*c_levels_pos[::-1]
    c_levels = np.concatenate([c_levels_neg, c_levels_pos])

    # 
    c_norm    = mpl.colors.Normalize(
                        vmin = c_min,
                        vmax = c_max)
    #            
    c_map     = plt.get_cmap(c_map_name)
    c_map.set_over('green')
    c_map.set_under('green')
    
    # Create the axes if necessary.
    if ax is None:

        ax  = plt.axes(projection = ccrs.Mollweide())
        ax.set_global()
    
    contour_info = {
        'lon_grid'  : lon_grid,
        'lat_grid'  : lat_grid,
        'c_levels'  : c_levels,
        'norm'      : c_norm,
        'cmap'      : c_map}

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
        arrows = ax.quiver(
            np.rad2deg(lon_grid[::slice_]),
            np.rad2deg(lat_grid[::slice_]),
            v_e[::slice_, ::slice_],
            v_n[::slice_, ::slice_],
            transform   = ccrs.PlateCarree(),
            scale       = 1.0E1,
            pivot       = 'middle',
            color       = 'mediumblue')

    else:

        arrows = None

    # Add the colour bar.
    if ax_cbar is not None:

        c_bar = plt.colorbar(
                    conts,
                    cax         = ax_cbar,
                    orientation = 'horizontal',
                    ticks = MultipleLocator(0.5))
        #c_bar.set_label('Magnitude of displacement', fontsize = 12)
        c_bar.set_label(c_bar_label, fontsize = 12)
    
    return conts, arrows, contour_info

def plot_sh_disp_3_comp(lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, ax_arr = None, fig = None, axes_grid_int = 111, show = True, title = None, n_c_levels = 21, c_bar_label = 'Displacement'): 
    '''
    Plots a vector field on the surface of a sphere in terms of the radial, consoidal and toroidal components.

    Input:

    lon_grid, lat_grid, U_r, V_e, V_n, W_e, W_n, show, title
        See 'Definitions of variables' in NMPostProcess/process.py.
    ax_arr
        An array of three GeoAxes, one for each component.
    n_c_levels
        See plot_sh_disp().
    '''
    
    # Create axis array.
    # It must have three axes and the third axis must have the cax (colorbar axis) attribute.
    if ax_arr is None:
        
        # Create the axes.
        #projection = ccrs.Mollweide()
        projection = ccrs.Robinson()
        if fig is None:

            fig = plt.figure(figsize = (3.5, 6.0))

        axes_class = (GeoAxes, dict(map_projection = projection))
        ax_arr = AxesGrid(fig, axes_grid_int,
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
    
    # Plot the displacement for each component.
    h_U_contour, _, contour_info = plot_sh_disp(lon_grid, lat_grid, v_r = U_r,             ax = ax_U, n_c_levels = n_c_levels)
    h_V_contour, h_V_arrows, _ = plot_sh_disp(lon_grid, lat_grid, v_e = V_e, v_n = V_n,  ax = ax_V, n_c_levels = n_c_levels)
    h_W_contour, h_W_arrows, _ = plot_sh_disp(lon_grid, lat_grid, v_e = W_e, v_n = W_n,  ax = ax_W, n_c_levels = n_c_levels, ax_cbar = ax_W.cax, c_bar_label = c_bar_label)

    handles = [h_U_contour, h_V_contour, h_V_arrows, h_W_contour, h_W_arrows]

    # Add a title (if requested).
    if title is not None:

        plt.suptitle(title)
    
    # Tidy around axes.
    plt.tight_layout()
    
    # Show (if requested).
    if show:
        
        plt.show()

    return fig, ax_arr, handles, contour_info

# Plot displacement animations. -----------------------------------------------
def plot_sh_disp_animation(dir_NM, i_mode, i_radius_str, n_lat_grid, mode_real_or_complex, rotation_period_hrs, save = True):

    # Determine if the plot is 'quick' mode or 'full' mode.
    if i_radius_str is None:

        option = 'quick'

    else:

        option = 'full'

    if mode_real_or_complex == 'real':

        raise NotImplementedError

    elif mode_real_or_complex in ['complex_real_only', 'complex_imag_only']:

        print('Animation for complex mode requires both real and imaginary parts. Change flag from \'{:}\' to \'complex\' in input_plotting.txt'.format(mode_real_or_complex))
        raise ValueError

    elif mode_real_or_complex == 'complex':

        # Plot the first frame, which is the real part.
        fig, ax_arr, handles, fields, contour_info = plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, 'complex_real_only', show = False, fmt = None, close = False, c_bar_label = 'Displacement', figsize = (3.5, 7.0))

        # Identify the eigenvalue list file.
        # Load the frequency of the mode.
        dir_processed = os.path.join(dir_NM, 'processed')
        path_eigenvalues = os.path.join(dir_processed, 'eigenvalue_list.txt')
        freq_mHz = read_eigenvalues(path_eigenvalues, i_mode = i_mode)
        freq_Hz = 1.0E-3*freq_mHz
    
        time_str_fmt = 'Time = {:>6.2f} h,\n{:>.2f} mode cycles,\n{:>.2f} rotations'
        time_str = time_str_fmt.format(0.0, 0.0, 0.0)
        time_title = ax_arr[0].set_title(time_str)

        rotation_period_seconds = 60.0*60.0*rotation_period_hrs
        t_min = 0.0
        t_max = 0.25*rotation_period_seconds
        n_t = 200
        t = np.linspace(t_min, t_max, n_t)
        
        period_mode = 1.0/freq_Hz
        t_hours = t/3600.0
        t_mode_cycles = t/period_mode
        t_rotations = t_hours/rotation_period_hrs

        omega = 2.0*np.pi*freq_Hz # Angular freq, radians per second.
        phase = omega*t
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        Ur_real = (fields['U']['real']['r']*cos_phase[:, np.newaxis, np.newaxis]) - (fields['U']['imag']['r']*sin_phase[:, np.newaxis, np.newaxis])
        Ur_imag = (fields['U']['real']['r']*sin_phase[:, np.newaxis, np.newaxis]) + (fields['U']['imag']['r']*cos_phase[:, np.newaxis, np.newaxis])

        Ve_real = (fields['V']['real']['e']*cos_phase[:, np.newaxis, np.newaxis]) - (fields['V']['imag']['e']*sin_phase[:, np.newaxis, np.newaxis])
        Ve_imag = (fields['V']['real']['e']*sin_phase[:, np.newaxis, np.newaxis]) + (fields['V']['imag']['e']*cos_phase[:, np.newaxis, np.newaxis])
        Vn_real = (fields['V']['real']['n']*cos_phase[:, np.newaxis, np.newaxis]) - (fields['V']['imag']['n']*sin_phase[:, np.newaxis, np.newaxis])
        Vn_imag = (fields['V']['real']['n']*sin_phase[:, np.newaxis, np.newaxis]) + (fields['V']['imag']['n']*cos_phase[:, np.newaxis, np.newaxis])

        We_real = (fields['W']['real']['e']*cos_phase[:, np.newaxis, np.newaxis]) - (fields['W']['imag']['e']*sin_phase[:, np.newaxis, np.newaxis])
        We_imag = (fields['W']['real']['e']*sin_phase[:, np.newaxis, np.newaxis]) + (fields['W']['imag']['e']*cos_phase[:, np.newaxis, np.newaxis])
        Wn_real = (fields['W']['real']['n']*cos_phase[:, np.newaxis, np.newaxis]) - (fields['W']['imag']['n']*sin_phase[:, np.newaxis, np.newaxis])
        Wn_imag = (fields['W']['real']['n']*sin_phase[:, np.newaxis, np.newaxis]) + (fields['W']['imag']['n']*cos_phase[:, np.newaxis, np.newaxis])

        def animate(i):
            
            time_title.set_text(time_str_fmt.format(t_hours[i], t_mode_cycles[i], t_rotations[i]))

            ax = ax_arr[0]
            
            v_scalar = Ur_real[i, :, :]
            v_scalar, lon_grid = add_cyclic_point(v_scalar, contour_info['lon_grid'])
            ax.collections = [] 
            ax.contourf(
                np.rad2deg(lon_grid),
                np.rad2deg(contour_info['lat_grid']),
                v_scalar,
                contour_info['c_levels'],
                transform   = ccrs.PlateCarree(),
                norm        = contour_info['norm'],
                cmap        = contour_info['cmap'])
            
            ax = ax_arr[1]
            # slice_ controls the down-sampling of the data.
            slice_ = int(np.ceil(lon_grid.shape[0]/20))
            
            v_e = Ve_real[i, :, :] 
            v_n = Vn_real[i, :, :]
            v_scalar = np.sqrt((v_e**2.0) + (v_n**2.0))
            v_scalar, lon_grid = add_cyclic_point(v_scalar, contour_info['lon_grid'])

            ax.collections = [] 
            ax.contourf(
                np.rad2deg(lon_grid),
                np.rad2deg(contour_info['lat_grid']),
                v_scalar,
                contour_info['c_levels'],
                transform   = ccrs.PlateCarree(),
                norm        = contour_info['norm'],
                cmap        = contour_info['cmap'], zorder = -1)

            #print(handles[2].__dict__)
            #handles[2]['U'] = v_e[::slice_, ::slice_]
            #handles[2]['V'] = v_n[::slice_, ::slice_]
            #handles[2].set_UVC(
            #    v_e[::slice_, ::slice_],
            #    v_n[::slice_, ::slice_],)

            slice_ = int(np.ceil(lon_grid.shape[0]/20))
            arrows = ax.quiver(
                np.rad2deg(lon_grid[::slice_]),
                np.rad2deg(contour_info['lat_grid'][::slice_]),
                v_e[::slice_, ::slice_],
                v_n[::slice_, ::slice_],
                transform   = ccrs.PlateCarree(),
                scale       = 1.0E1,
                pivot       = 'middle',
                color       = 'mediumblue')

            ax = ax_arr[2]
            # slice_ controls the down-sampling of the data.
            slice_ = int(np.ceil(lon_grid.shape[0]/20))
            
            v_e = We_real[i, :, :] 
            v_n = Wn_real[i, :, :]
            v_scalar = np.sqrt((v_e**2.0) + (v_n**2.0))
            v_scalar, lon_grid = add_cyclic_point(v_scalar, contour_info['lon_grid'])

            ax.collections = [] 
            ax.contourf(
                np.rad2deg(lon_grid),
                np.rad2deg(contour_info['lat_grid']),
                v_scalar,
                contour_info['c_levels'],
                transform   = ccrs.PlateCarree(),
                norm        = contour_info['norm'],
                cmap        = contour_info['cmap'], zorder = -1)

            #print(handles[2].__dict__)
            #handles[2]['U'] = v_e[::slice_, ::slice_]
            #handles[2]['V'] = v_n[::slice_, ::slice_]
            #handles[2].set_UVC(
            #    v_e[::slice_, ::slice_],
            #    v_n[::slice_, ::slice_],)

            slice_ = int(np.ceil(lon_grid.shape[0]/20))
            arrows = ax.quiver(
                np.rad2deg(lon_grid[::slice_]),
                np.rad2deg(contour_info['lat_grid'][::slice_]),
                v_e[::slice_, ::slice_],
                v_n[::slice_, ::slice_],
                transform   = ccrs.PlateCarree(),
                scale       = 1.0E1,
                pivot       = 'middle',
                color       = 'mediumblue')


        anim = FuncAnimation(
            fig, animate, interval= 10, frames=len(t)-1)

        fmt = 'gif'
        if option == 'quick':

            name_anim = 'disp_anim_{:}_{:>05d}.{:}'.format(option, i_mode, fmt)

        else:

            name_anim = 'disp_anim_{:}_{:>05d}_{:>03d}.{:}'.format(option, i_mode, j_radius, fmt)

        if save:

            dir_plot = os.path.join(dir_processed, 'plots')
            path_anim = os.path.join(dir_plot, name_anim)
            print('Saving animation to {:}'.format(path_anim))
            anim.save(path_anim, writer='imagemagick')

        plt.show()

    else:

        raise ValueError

    return

# Plot real, imaginary, absolute and phase values of the complex radial component.
def plot_radial_cmplx_wrapper(dir_NM, i_mode, n_lat_grid, i_radius_str = None, show = True, fmt = 'png', transparent = True):

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)

    # Determine if the plot is 'quick' mode or 'full' mode.
    if i_radius_str is None:

        option = 'quick'

    else:

        option = 'full'

    fig = plt.figure(figsize = (7.0, 5.0))

    # Reconstruct the coordinate grid.
    n_lon_grid = (2*n_lat_grid) - 1
    lon_grid = np.linspace(0.0, 2.0*np.pi, n_lon_grid + 1, endpoint = True)[:-1]
    lat_grid = np.linspace(-np.pi/2.0, np.pi/2.0, n_lat_grid, endpoint=True)

    # List of radii to sample.
    if i_radius_str == 'all':

        _, _, _, _, _, header_info = load_vsh_coefficients(dir_NM, i_mode, 0)
        n_samples = len(header_info['r_sample'])

        i_radius_list = list(range(n_samples))

    elif i_radius_str is None:

        i_radius_list = [None]

    else:

        i_radius_list = [int(i_radius_str)]

    first_iteration = True
    for j_radius in i_radius_list:
        
        # Load the VSH coefficients for the specified mode.
        coeffs, header_info, r_sample, i_sample = load_vsh_coefficients(dir_NM, i_mode, j_radius)
        Ulm_real, Vlm_real, Wlm_real, Ulm_imag, Vlm_imag, Wlm_imag = coeffs

        title = region_int_to_title(i_sample, r_sample, shell_name_path = os.path.join(dir_processed, 'shell_names.txt'))

        if first_iteration:

            # Infer the maximum l-value used.
            #n_coeffs = len(Ulm)
            n_coeffs = coeffs.shape[-1]
            l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

            # Re-construct the shtns calculator.
            m_max = l_max
            sh_calculator   = shtns.sht(l_max, m_max)
            grid_type       =       shtns.sht_reg_fast          \
                                |   shtns.SHT_SOUTH_POLE_FIRST  \
                                |   shtns.SHT_PHI_CONTIGUOUS
            sh_calculator.set_grid(n_lat_grid, n_lon_grid, flags = grid_type)

            first_iteration = False

        n_c_levels_default = 20 

        # Expand the VSH into spatial components and plot.
        U_real_r, V_real_e, V_real_n, W_real_e, W_real_n = project_from_spherical_harmonics(sh_calculator, Ulm_real, Vlm_real, Wlm_real)
        U_imag_r, V_imag_e, V_imag_n, W_imag_e, W_imag_n = project_from_spherical_harmonics(sh_calculator, Ulm_imag, Vlm_imag, Wlm_imag)

        U_cplx_r = U_real_r + 1.0j*U_imag_r
        #U_abs_r = np.abs(U_cplx_r)
        #U_phase_r = np.angle(U_cplx_r)

        plot_radial_cplx(lon_grid, lat_grid, U_cplx_r, fig = fig, ax_arr = None, show = False, title = title, n_c_levels = n_c_levels_default)

        # Save the plot.
        fig_base_name = 'disp_radial_cplx'
        if option == 'quick':

            name_fig = '{:}_{:}_{:>05d}.{:}'.format(fig_base_name, option, i_mode, fmt)

        else:

            name_fig = '{:}_{:}_{:>05d}_{:>03d}.{:}'.format(fig_base_name, option, i_mode, j_radius, fmt)

        path_fig = os.path.join(dir_plot, name_fig)
        print('Saving figure to {:}'.format(path_fig))
        save_figure(path_fig, fmt, transparent = transparent)

        # Show the plot.
        if show:

            plt.show()

        # Close the plot.
        plt.close()

    return

def plot_radial_cplx(lon_grid, lat_grid, U_r, fig = None, ax_arr = None, show = False, title = None, n_c_levels = 21):

    # Create axis array.
    # It must have three axes and the third axis must have the cax (colorbar axis) attribute.
    if ax_arr is None:
        
        # Create the axes.
        #projection = ccrs.Mollweide()
        projection = ccrs.Robinson()
        if fig is None:

            fig = plt.figure(figsize = (7.0, 5.0))

        axes_class = (GeoAxes, dict(map_projection = projection))
        ax_arr = AxesGrid(fig, 111,
                axes_class = axes_class,
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                #cbar_mode='single',
                #cbar_location='bottom',
                label_mode = '',
                #cbar_pad=0.1,
                )

    # Plot the displacement for each component.
    plot_sh_disp(lon_grid, lat_grid, v_r = np.real(U_r),    ax = ax_arr[0], n_c_levels = n_c_levels)
    plot_sh_disp(lon_grid, lat_grid, v_r = np.imag(U_r),    ax = ax_arr[1], n_c_levels = n_c_levels)
    plot_sh_disp(lon_grid, lat_grid, v_r = np.abs(U_r),     ax = ax_arr[2], n_c_levels = n_c_levels)
    plot_sh_disp(lon_grid, lat_grid, v_r = np.angle(U_r),   ax = ax_arr[3], n_c_levels = n_c_levels, c_lims = [-np.pi, np.pi], c_map_name = 'twilight')

    # Add a title (if requested).
    if title is not None:

        plt.suptitle(title)
    
    # Tidy around axes.
    plt.tight_layout()
    
    # Show (if requested).
    if show:
        
        plt.show()

    return

# Plot spectral data. ---------------------------------------------------------
def plot_sh_real_coeffs_3_comp_all_modes(dir_NM, fmt = 'pdf', i_radius_str = None):
    '''
    Loop over all the modes and plot spherical harmonics.
    A wrapper for plot_sh_real_coeffs_3_comp_wrapper().
    '''

    #mode_list = get_list_of_modes_from_output_files(dir_NM)
    if i_radius_str is None:

        option = 'quick'

    else:

        option = 'full'

    mode_list = get_list_of_modes_from_coeff_files(dir_NM, option)
    
    for i_mode in mode_list:

        print('\nPlotting spectrum for mode {:>5d}.'.format(i_mode))

        plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, show = False, fmt = fmt, i_radius_str = i_radius_str)

    return

def plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, show = True, fmt = 'pdf', i_radius_str = None):
    '''
    Load coefficients, then plot 3 sets of spherical harmonic coefficients on a grid.
    A wrapper for plot_sh_real_coeffs_3_comp().
    '''

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_plot = os.path.join(dir_processed, 'plots')
    mkdir_if_not_exist(dir_plot)

    # Determine if the plot is 'quick' mode or 'full' mode.
    if i_radius_str is None:

        option = 'quick'

    else:

        option = 'full'

    if i_radius_str == 'all':

        _, header_info, _, _ = load_vsh_coefficients(dir_NM, i_mode, 0)
        n_samples = len(header_info['r_sample'])

        i_radius_list = list(range(n_samples))

    elif i_radius_str is None:

        i_radius_list = [None]

    else:

        i_radius_list = [int(i_radius_str)]

    fixed_scale = False
    if fixed_scale:

        c_max = 0.3

    else:

        c_max = None

    first_iteration = True
    for j_radius in i_radius_list:

        # Load VSH coefficients.
        #Ulm, Vlm, Wlm, r_sample, i_sample, _ = load_vsh_coefficients(dir_NM, i_mode, j_radius)
        coeffs, _, r_sample, i_sample = load_vsh_coefficients(dir_NM, i_mode, j_radius)
        Ulm, Vlm, Wlm = coeffs
        
        first_iteration = True
        if first_iteration:

            # Infer the maximum l-value used.
            n_coeffs = len(Ulm)
            l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

            first_iteration = False

        # Get title.
        title = region_int_to_title(i_sample, r_sample, shell_name_path = os.path.join(dir_processed, 'shell_names.txt'))
        
        # Convert from complex to real form.
        ulm, l, m = convert_complex_sh_to_real(Ulm, l_max)
        vlm, _, _ = convert_complex_sh_to_real(Vlm, l_max)
        wlm, _, _ = convert_complex_sh_to_real(Wlm, l_max)

        # Multiply V and W by k.
        k = np.sqrt(l*(l + 1.0))
        vlm = k*vlm
        wlm = k*wlm

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
                c_max       = c_max,
                abs_plot    = False)

        # Save the plot.
        if option == 'quick':

            name_fig = 'spectrum_quick_{:>05d}.{:}'.format(i_mode, fmt)

        elif option == 'full':

            name_fig = 'spectrum_full_{:>05d}_{:>03d}.{:}'.format(i_mode, j_radius, fmt)

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
    '''
    Plot spherical harmonics on a grid.
    '''
    
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

    # Tidy up the ticks.
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
    '''
    Plot three sets of spherical harmonics on a grid.
    A wrapper for plot_sh_real_coeffs().
    '''

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
    dir_PM, dir_NM, _, l_max, i_mode_str, n_radii = read_input_NMPostProcess()

    # Read the input_plotting file.
    option, i_radius_str, plot_type, i_mode_str, fmt, n_lat_grid, \
    mode_real_or_complex, rotation_period_hrs = read_input_plotting()

    print(option, i_radius_str, plot_type, i_mode_str, fmt, n_lat_grid, mode_real_or_complex, rotation_period_hrs)

    # Decide whether to show the plots.
    if i_mode_str == 'all' or i_radius_str == 'all':

        show = False

    else:

        show = True

    # Plot all modes.
    if i_mode_str == 'all':

        # Spatial plot.
        if plot_type == 'spatial':

            plot_sh_disp_all_modes(dir_NM, n_lat_grid, mode_real_or_complex, fmt = fmt, i_radius_str = i_radius_str)

        # Spectral plot.
        elif plot_type == 'spectral':

            plot_sh_real_coeffs_3_comp_all_modes(dir_NM, fmt = fmt, i_radius_str = i_radius_str)

        elif plot_type == 'animation':

            raise ValueError('Cannot plot animation for all modes simultaneously.')

        else:

            raise ValueError('Plot type {:} from input file not recognised.'.format(plot_type))

    # Plot one mode.
    else:

        i_mode = int(i_mode_str)
        if plot_type == 'spatial':

            plot_sh_disp_wrapper(dir_NM, i_mode, n_lat_grid, mode_real_or_complex, fmt = fmt, i_radius_str = i_radius_str, show = show)

        elif plot_type == 'spectral':
            
            plot_sh_real_coeffs_3_comp_wrapper(dir_NM, i_mode, fmt = fmt, i_radius_str = i_radius_str, show = show)

        elif plot_type == 'radial_complex':
            
            plot_radial_cmplx_wrapper(dir_NM, i_mode, n_lat_grid, i_radius_str = None)

        elif plot_type == 'animation':

            plot_sh_disp_animation(dir_NM, i_mode, i_radius_str, n_lat_grid, mode_real_or_complex, rotation_period_hrs)

        else:

            raise ValueError('Plot type {:} from input file not recognised.'.format(plot_type))

if __name__ == '__main__':

    main()
