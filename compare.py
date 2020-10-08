import os

import numpy as np

from common import convert_complex_sh_to_real, load_all_vsh_coefficients, read_input_NMPostProcess, read_eigenvalues

def read_mode_info(dir_NM):

    # Define directories and files.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Load the mode identification information.
    path_ids = os.path.join(dir_processed, 'mode_ids_quick.txt')
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)

    # Read coefficients.
    coeffs = load_all_vsh_coefficients(dir_NM, i_mode)

    return i_mode, f, l, type_, shell, coeffs 

def coeff_match(coeffs_1, coeffs_2):

    coeffs_1_flat = coeffs_1.flatten()
    coeffs_2_flat = coeffs_2.flatten()

    coeffs_1_norm = np.sqrt(np.sum(coeffs_1_flat**2.0))
    coeffs_2_norm = np.sqrt(np.sum(coeffs_1_flat**2.0))

    coeffs_dot = np.sum(coeffs_1*coeffs_2)
    coeffs_dot_normalised = coeffs_dot/(coeffs_1_norm*coeffs_2_norm)

    #print(coeffs_1.shape, coeffs_2.shape)
    #print(coeffs_1_norm)
    #print(coeffs_2_norm)

    return coeffs_dot_normalised

def main():

    # Read the input file and data of the first run.
    dir_PM_1, dir_NM_1, _, l_max_1, _, _ = \
        read_input_NMPostProcess()
    i_mode_1, f_1, l_1, type_1, shell_1, coeffs_cplx_1 = \
        read_mode_info(dir_NM_1)

    # Read the data of the second run.
    file_compare = 'input_compare.txt'
    with open(file_compare, 'r') as in_id:

        dir_NM_2 = in_id.readline().strip()

    # Read the data of the second run.
    i_mode_2, f_2, l_2, type_2, shell_2, coeffs_cplx_2 = \
        read_mode_info(dir_NM_2)

    # Convert the coefficients from complex to real.
    n_mode_1 = coeffs_cplx_1.shape[0]
    n_mode_2 = coeffs_cplx_2.shape[0]
    l_max = l_max_1
    n_real_coeffs = (l_max + 1)*(l_max + 1)
    coeffs_real_1 = np.zeros((n_mode_1, 3, n_real_coeffs))
    coeffs_real_2 = np.zeros((n_mode_2, 3, n_real_coeffs))
    for i in range(n_mode_1):

        for j in range(3):
            
            coeffs_real_1[i, j, :], _, _ = convert_complex_sh_to_real(
                                        coeffs_cplx_1[i, j, :], l_max)

    for i in range(n_mode_2):

        for j in range(3):

            coeffs_real_2[i, j, :], _, _ = convert_complex_sh_to_real(
                                        coeffs_cplx_2[i, j, :], l_max)

    # Find the best match for each mode in the first list by comparison
    # with the coefficients in the second list.
    f_thresh = 0.1 # mHz.
    best_match = np.zeros(n_mode_1, dtype = np.int)
    for i in range(n_mode_1):

        # Narrow down the modes for comparison by using the frequency
        # threshold.
        j_thresh_match = np.where((np.abs(f_1[i] - f_2) < f_thresh))[0]

        # For each mode matching the frequency, compare the coefficients.
        n_thresh_match = len(j_thresh_match)
        fit = np.zeros(n_thresh_match)

        for k, j in enumerate(j_thresh_match):

            fit[k] = coeff_match(coeffs_real_1[i, ...], coeffs_real_2[j, ...])

        best_match[i] = j_thresh_match[np.argmax(np.abs(fit))]

    print('Best-matching modes:')
    for i in range(n_mode_1):

        print('{:>5d} {:>5d}'.format(i, best_match[i]))

    path_out = os.path.join(dir_NM_1, 'processed', 'comparison.txt')
    print('Saving match information to {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in range(n_mode_1):

            out_id.write('{:>5d} {:>5d}\n'.format(i, best_match[i]))

    return

if __name__ == '__main__':

    main()
