import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from NMPostProcess.common import convert_complex_sh_to_real, load_vsh_coefficients, read_discon_file, read_eigenvalues, read_input_NMPostProcess, read_input_plotting
from NMPostProcess.plot.plot_common import save_figure

try:
    from Ouroboros.common import (load_eigenfreq, load_eigenfunc, read_input_file)
except ImportError:
    print("Warning: Could not import Ouroboros. Comparison with 1-D eigenfunctions will not be available.")

def get_radial_profiles_real(coeffs):

    # Extract the components.
    Ulm = coeffs[:, 0, :]
    Vlm = coeffs[:, 1, :]
    Wlm = coeffs[:, 2, :]
    
    # Get number of radii, number of coefficients and l_max. 
    n_radii     = Ulm.shape[0]
    n_imag_coeffs    = Ulm.shape[1]
    l_max = (int((np.round(np.sqrt(8*n_imag_coeffs + 1)) - 1))//2) - 1

    # Prepare real output arrays.
    n_real_coeffs = (l_max + 1)**2 
    ulm = np.zeros((n_radii, n_real_coeffs))
    vlm = np.zeros((n_radii, n_real_coeffs))
    wlm = np.zeros((n_radii, n_real_coeffs))
    
    # Loop over radii.
    for i in range(n_radii):
        
        # Convert from complex to real form.
        ulm[i, :], l, m = convert_complex_sh_to_real(Ulm[i, :], l_max)
        vlm[i, :], _, _ = convert_complex_sh_to_real(Vlm[i, :], l_max)
        wlm[i, :], _, _ = convert_complex_sh_to_real(Wlm[i, :], l_max)

    xlm = np.concatenate([ulm, vlm, wlm])

    # Use SVD to extract first principal component.
    # Note: It is automatically normalised.
    _, _, PCs = np.linalg.svd(xlm)
    first_PC = PCs[0, :]

    # Take the dot product with the first PC to get the radial variation.
    U = np.zeros(n_radii)
    V = np.zeros(n_radii)
    W = np.zeros(n_radii)
    for i in range(n_radii):

        U[i] = np.sum(first_PC*ulm[i, :])
        V[i] = np.sum(first_PC*vlm[i, :])
        W[i] = np.sum(first_PC*wlm[i, :])

    ## Use SVD to extract first principal component.
    ## Note: The first P.C. is automatically normalised.
    #_, _, U_PCs = np.linalg.svd(ulm)
    #U_first_PC = U_PCs[0, :]
    ## 
    #_, _, V_PCs = np.linalg.svd(vlm)
    #V_first_PC = V_PCs[0, :]
    ##
    #_, _, W_PCs = np.linalg.svd(wlm)
    #W_first_PC = W_PCs[0, :]

    ## Take the dot product with the first PC to get the radial variation.
    #U = np.zeros(n_radii)
    #V = np.zeros(n_radii)
    #W = np.zeros(n_radii)
    #for i in range(n_radii):

    #    U[i] = np.sum(U_first_PC*ulm[i, :])
    #    V[i] = np.sum(V_first_PC*vlm[i, :])
    #    W[i] = np.sum(W_first_PC*wlm[i, :])

    return U, V, W

def get_radial_profiles_complex(coeffs):

    n_radii = coeffs.shape[0]

    # Extract the components.
    Ulm = coeffs[:, 0, :] + 1.0j*coeffs[:, 3, :]
    Vlm = coeffs[:, 1, :] + 1.0j*coeffs[:, 4, :]
    Wlm = coeffs[:, 2, :] + 1.0j*coeffs[:, 5, :]

    Xlm = np.concatenate([Ulm, Vlm, Wlm])

    # Use SVD to extract first principal component.
    # Note: It is automatically normalised.
    _, _, PCs = np.linalg.svd(Xlm)
    first_PC = PCs[0, :]

    ## Note: Already normalised.
    #norm_first_PC = np.sqrt(np.sum(first_PC * np.conj(first_PC)))

    # Take the dot product with the first PC to get the radial variation.
    U = np.zeros(n_radii)
    V = np.zeros(n_radii)
    W = np.zeros(n_radii)
    first_PC_conj = np.conjugate(first_PC)
    for i in range(n_radii):
        
        U[i] = np.sum(np.real(first_PC_conj * Ulm[i, :]))
        V[i] = np.sum(np.real(first_PC_conj * Vlm[i, :]))
        W[i] = np.sum(np.real(first_PC_conj * Wlm[i, :]))

    return U, V, W

def check_sign_radial_profiles(r, U, V, W):

    # Find 'dominant' eigenfunction.
    X = np.array([U, V, W])
    max_X = np.max(np.abs(X), axis = 1)
    i_dom = np.argmax(max_X)

    # Find sign of dominant eigenfunction at maximum absolute value.
    j_max = np.argmax(np.abs(X[i_dom, :]))
    sign_max = np.sign(X[i_dom, j_max])
    X_max = np.abs(X[i_dom, j_max])

    # Multiply eigenfunctions by sign.
    X = X*sign_max
    U, V, W = X

    return U, V, W, X_max

def load_compare_eigfunc(compare_info):

    run_info = read_input_file(compare_info['path'])

    # Get frequency information.
    mode_info = load_eigenfreq(run_info, compare_info['type'],
                    n_q = compare_info['n'], l_q = compare_info['l'])
    f = mode_info['f']

    # Get normalisation arguments.
    f_rad_per_s = f*1.0E-3*2.0*np.pi
    norm_func = 'DT'
    norm_units = 'SI'
    normalisation_args = {'norm_func' : norm_func, 'units' : norm_units}
    normalisation_args['omega'] = f_rad_per_s

    # Get eigenfunction information.
    eigfunc_dict = load_eigenfunc(run_info, compare_info['type'],
                                compare_info['n'],
                                compare_info['l'],
                                norm_args = normalisation_args)
    eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.

    return eigfunc_dict

def get_rms_amp(r, U, V, W):
    
    if r[-1] < r[0]:
        r = r[::-1]
        U = U[::-1]
        V = V[::-1]
        W = W[::-1]

    func = U**2.0 + V**2.0 + W**2.0
    #r_range = np.max(r) - np.min(r)
    #rms_amp = np.sqrt(np.trapz(func, x = r)/r_range)
    rms_amp = np.sqrt(np.trapz(func, x = r)/np.max(r))

    return rms_amp

def plot_eigenfunction_wrapper(dir_PM, dir_NM, i_mode, fmt = 'png', mode_real_or_complex = 'real', multiply_by_k = True, compare_info = None, show = True):

    # Load comparison eigenfunction.
    if compare_info is not None:

        eigfunc_dict = load_compare_eigfunc(compare_info)

    # Load the mode frequency.
    dir_processed           = os.path.join(dir_NM, 'processed')
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    i_mode_list, freq_list  = read_eigenvalues(file_eigval_list)
    i_mode_list             = np.array(i_mode_list, dtype = np.int)
    freq_list               = np.array(freq_list)
    freq                    = freq_list[np.where(i_mode_list == i_mode)[0][0]]

    # Create the title.
    title = 'Mode {:>5d}, frequency {:>7.3f} mHz'.format(i_mode, freq)

    # Read the discontinuity radius information.
    path_discon_info = os.path.join(dir_PM, 'radii.txt')
    r_discons, state_outer = read_discon_file(path_discon_info)
    n_discons = len(r_discons)

    # Load the coefficients and radial coordinates.
    coeffs, header, _, _ = load_vsh_coefficients(dir_NM, i_mode, i_radius = 'all')
    r = header['r_sample']

    # Calculate radial profiles from the coefficients.
    if mode_real_or_complex == 'real':

        U, V, W = get_radial_profiles_real(coeffs)

    elif mode_real_or_complex == 'complex':

        U, V, W = get_radial_profiles_complex(coeffs)

    # 
    if multiply_by_k:

        # Load the mode identification information.
        path_ids = os.path.join(dir_processed, 'mode_ids_full.txt')
        _, l_list, _, _ = np.loadtxt(path_ids, dtype = np.int).T
        l = l_list[i_mode - 1] # Not 0- and 1-based indexing.
        k = np.sqrt(l*(l + 1))

        V = k*V
        W = k*W

    # Adjust sign according to convention.
    U, V, W, X_max = check_sign_radial_profiles(r, U, V, W)

    fig = plt.figure(figsize = (6.0, 8.0))
    ax = plt.gca()
    
    if compare_info is None:

        labels = ['U', 'V', 'W']

    else:

        labels = ['{:} (NormalModes)'.format(x) for x in ['U', 'V', 'W']]

    arrays = [U, V, W]
    color_U = 'orangered'
    color_V = 'royalblue'
    color_W = 'mediumseagreen'
    colors = [color_U, color_V, color_W]

    for i in range(3):

        ax.plot(arrays[i], r, label = labels[i], color = colors[i])
        ax.scatter(arrays[i], r, color = colors[i], s = 3)

    if compare_info is not None:
        
        if compare_info['type'] == 'S':

            # Adjust sign according to convention.
            #eigfunc_dict['U'], eigfunc_dict['V'], _, _ = \
            #    check_sign_radial_profiles(eigfunc_dict['r'],
            #        eigfunc_dict['U'], eigfunc_dict['V'],
            #        np.zeros(len(eigfunc_dict['r'])))
            if np.sign(eigfunc_dict['U'][-1]) != np.sign(U[0]):

                eigfunc_dict['U'] = -1.0*eigfunc_dict['U']
                eigfunc_dict['V'] = -1.0*eigfunc_dict['V']

            rms_amp = get_rms_amp(r, U, V, W)
            rms_compare = get_rms_amp(eigfunc_dict['r'],
                            eigfunc_dict['U'], eigfunc_dict['V'],
                            np.zeros(len(eigfunc_dict['r'])))
            rms_ratio = rms_amp/rms_compare
            
            colors = [color_U, color_V]
            components = ['U', 'V']

            for key in components:
                
                eigfunc_dict[key] = eigfunc_dict[key]*rms_ratio

        else:

            raise NotImplementedError

        #labels = ['{:} (Ouroboros)'.format(x) for x in components]
        labels = ['{:} (Mineos)'.format(x) for x in components]
        arrays = [eigfunc_dict[key] for key in components] 

        n_components = len(components)
        for i in range(n_components):
            
            ax.plot(arrays[i], eigfunc_dict['r'], color = colors[i],
                    linestyle = ':', label = labels[i])

    # Set axis limits.
    x_buff = 1.1
    ax.set_xlim([-x_buff*X_max, x_buff*X_max])
    ax.set_ylim([0.0, r_discons[0]])

    # Draw guidelines.
    guideline_kwargs = {'linestyle' : ':', 'color' : 'k'}
    ax.axvline(**guideline_kwargs)

    if n_discons > 1:

        for i in range(1, n_discons):

            ax.axhline(r_discons[i], **guideline_kwargs)

    ax.legend()

    font_size_label = 12
    font_size_title = 16
    ax.set_xlabel('Eigenfunction', fontsize = font_size_label)
    ax.set_ylabel('Radius / km', fontsize = font_size_label)
    ax.set_title(title, fontsize = font_size_title)

    plt.tight_layout()

    # Save the figure.
    dir_plot = os.path.join(dir_processed, 'plots')
    name_fig = 'eigenfunctions_{:>05d}.{:}'.format(i_mode, fmt)
    path_fig = os.path.join(dir_plot, name_fig)
    print('Saving figure to {:}'.format(path_fig))
    save_figure(path_fig, fmt)
    
    if show:

        plt.show()

    plt.close()

    return

def main():

    # Read input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_compare", nargs = 4, metavar = ('path_input', 'mode_type', 'n', 'l'), help = "Path to Ouroboros input file and mode ID for comparison of eigenfunction.")
    args = parser.parse_args()
    if args.path_compare is not None:
        
        compare_info = dict()
        compare_info['path'], compare_info['type'], compare_info['n'], compare_info['l'] = \
            args.path_compare

        compare_info['n'] = int(compare_info['n'])
        compare_info['l'] = int(compare_info['l'])

    else:

        compare_info = None

    # Read the NMPostProcess input file.
    dir_PM, dir_NM, _, _, _, _ = read_input_NMPostProcess()

    # Read the input_plotting file.
    option, i_radius_str, plot_type, i_mode_str, fmt, n_lat_grid, \
    mode_real_or_complex, rotation_period_hrs = read_input_plotting()

    if i_mode_str == 'all':

        raise NotImplementedError

    else:

        #i_mode = int(i_mode_str)
        for i_mode in range(85, 116):

            plot_eigenfunction_wrapper(dir_PM, dir_NM, i_mode, fmt = fmt,
                    mode_real_or_complex = mode_real_or_complex,
                    compare_info = compare_info,
                    show = False)

    return

if __name__ == '__main__':

    main()
