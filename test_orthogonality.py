import os

import numpy as np

from characterise import calculate_weights
from common import convert_complex_sh_to_real, load_vsh_coefficients, load_all_vsh_coefficients, make_l_and_m_lists, read_input_NMPostProcess, read_eigenvalues
from compare import coeff_match, read_mode_info

def compare_two_modes(dir_NM_a, i_a, dir_NM_b, i_b, l_max):

    # Read coefficients.
    coeffs_cplx_1_a, header_info, _, _ = load_vsh_coefficients(dir_NM_a, i_a, i_radius = 'all') 
    coeffs_cplx_1_b,           _, _, _ = load_vsh_coefficients(dir_NM_b, i_b, i_radius = 'all') 
    #coeffs_cplx_2 = load_vsh_coefficients(dir_NM_2, i_mode_2, option = option)

    # Calculate weights of each shell.
    volume_weights = calculate_weights(header_info['r_sample'])

    # Apply the weights.
    coeffs_cplx_1_a = coeffs_cplx_1_a*volume_weights[:, np.newaxis, np.newaxis]
    coeffs_cplx_1_b = coeffs_cplx_1_b*volume_weights[:, np.newaxis, np.newaxis]
    #coeffs_cplx_2 = coeffs_cplx_2*volume_weights[:, np.newaxis, np.newaxis]

    # Count the number of radii.
    n_radii = coeffs_cplx_1_a.shape[0]

    # Scale the coefficients by the k values.
    # First, get the list of l and m values of the coefficients.
    l, m = make_l_and_m_lists(l_max)
    k = np.sqrt(l*(l + 1.0))
    coeffs_cplx_1_a = coeffs_cplx_1_a*k
    coeffs_cplx_1_b = coeffs_cplx_1_b*k

    n_real_coeffs = (l_max + 1)*(l_max + 1)
    coeffs_real_1_a = np.zeros((n_radii, 3, n_real_coeffs))
    coeffs_real_1_b = np.zeros((n_radii, 3, n_real_coeffs))
    for j in range(n_radii):

        for k in range(3):

            coeffs_real_1_a[j, k, :], _, _ = convert_complex_sh_to_real(coeffs_cplx_1_a[j, k, :], l_max)
            coeffs_real_1_b[j, k, :], _, _ = convert_complex_sh_to_real(coeffs_cplx_1_b[j, k, :], l_max)
    
    fit = np.abs(coeff_match(coeffs_real_1_a, coeffs_real_1_b))

    str_a = dir_NM_a.split('/')[-2]
    str_b = dir_NM_b.split('/')[-2]

    print('Match between {:}-{:>05d} and {:}-{:>05d}: {:>5.1f} %'.format(str_a, i_a, str_b, i_b, fit*100.0))

    return

def main():

    # Read the input file and data of the first run.
    i_clip = None
    dir_PM_1, dir_NM_1, option, l_max_1, _, _ = \
        read_input_NMPostProcess()
    i_mode_1, f_1, l_1, type_1, shell_1 = \
        read_mode_info(dir_NM_1, option, i_clip = i_clip)
    dir_processed = os.path.join(dir_NM_1, 'processed')

    # Read the data of the second run.
    file_compare = 'input_compare.txt'
    with open(file_compare, 'r') as in_id:

        dir_NM_2 = in_id.readline().strip()

    # Read the data of the second run.
    i_mode_2, f_2, l_2, type_2, shell_2 = \
        read_mode_info(dir_NM_2, option, i_clip = i_clip)

    l_max = l_max_1
    compare_two_modes(dir_NM_1, 55, dir_NM_1,  56, l_max)
    compare_two_modes(dir_NM_1, 55, dir_NM_2, 106, l_max)
    compare_two_modes(dir_NM_1, 56, dir_NM_2, 106, l_max)

    return

if __name__ == '__main__':

    main()
