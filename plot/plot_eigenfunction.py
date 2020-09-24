import matplotlib.pyplot as plt
import numpy as np

from common import convert_complex_sh_to_real, load_vsh_coefficients, read_input_NMPostProcess, read_input_plotting

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

    # Use SVD to extract first principal component.
    _, _, U_PCs = np.linalg.svd(ulm)
    U_first_PC = U_PCs[0, :]
    # 
    _, _, V_PCs = np.linalg.svd(vlm)
    V_first_PC = V_PCs[0, :]
    #
    _, _, W_PCs = np.linalg.svd(wlm)
    W_first_PC = W_PCs[0, :]

    # Take the dot product with the first PC to get the radial variation.
    U = np.zeros(n_radii)
    V = np.zeros(n_radii)
    W = np.zeros(n_radii)
    for i in range(n_radii):

        U[i] = np.sum(U_first_PC*ulm[i, :])
        V[i] = np.sum(V_first_PC*vlm[i, :])
        W[i] = np.sum(W_first_PC*wlm[i, :])

    return U, V, W

def plot_eigenfunction_wrapper(dir_NM, i_mode, fmt = 'png'):

    # Load the coefficients.
    coeffs, header = load_vsh_coefficients(dir_NM, i_mode, i_radius = 'all')
    
    # Calculate radial profiles from the coefficients.
    U, V, W = get_radial_profiles(coeffs)

    print(header)
    r = header['r_sample']

    fig = plt.figure(figsize = (6.0, 8.0))
    ax = plt.gca()

    labels = ['U', 'V', 'W']
    arrays = [U, V, W]
    colors = ['r', 'g', 'b']

    for i in range(3):

        ax.plot(arrays[i], r, label = labels[i], color = colors[i])

    ax.legend()

    plt.show()


    #plt.scatter(np.real(vh[0, :]), np.real(U[1, :]))
    #plt.scatter(np.imag(vh[0, :]), np.imag(U[1, :]))

    #plt.show()

    return

def main():

    # Read the NMPostProcess input file.
    _, dir_NM, _, _, _, _ = read_input_NMPostProcess()

    # Read the input_plotting file.
    _, _, i_mode_str, fmt, _ = read_input_plotting()

    if i_mode_str == 'all':

        raise NotImplementedError

    else:

        i_mode = int(i_mode_str)
        plot_eigenfunction_wrapper(dir_NM, i_mode, fmt = fmt)

    return

if __name__ == '__main__':

    main()
