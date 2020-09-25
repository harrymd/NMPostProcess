import os

import matplotlib.pyplot as plt
import numpy as np

from common import convert_complex_sh_to_real, load_vsh_coefficients, read_discon_file, read_eigenvalues, read_input_NMPostProcess, read_input_plotting
from plot.plot_common import save_figure

def get_radial_profiles(coeffs):

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

def plot_eigenfunction_wrapper(dir_PM, dir_NM, i_mode, fmt = 'png'):

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
    coeffs, header = load_vsh_coefficients(dir_NM, i_mode, i_radius = 'all')
    r = header['r_sample']
    
    # Calculate radial profiles from the coefficients.
    U, V, W = get_radial_profiles(coeffs)

    # Adjust sign according to convention.
    U, V, W, X_max = check_sign_radial_profiles(r, U, V, W)

    fig = plt.figure(figsize = (6.0, 8.0))
    ax = plt.gca()

    labels = ['U', 'V', 'W']
    arrays = [U, V, W]
    colors = ['orangered', 'royalblue', 'mediumseagreen']

    for i in range(3):

        ax.plot(arrays[i], r, label = labels[i], color = colors[i])
        ax.scatter(arrays[i], r, color = colors[i], s = 3)

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

    plt.show()

    return

def main():

    # Read the NMPostProcess input file.
    dir_PM, dir_NM, _, _, _, _ = read_input_NMPostProcess()

    # Read the input_plotting file.
    _, _, _, i_mode_str, fmt, _ = read_input_plotting()

    if i_mode_str == 'all':

        raise NotImplementedError

    else:

        i_mode = int(i_mode_str)
        plot_eigenfunction_wrapper(dir_PM, dir_NM, i_mode, fmt = fmt)

    return

if __name__ == '__main__':

    main()
