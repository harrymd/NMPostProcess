import os

import numpy as np

from characterise import calculate_weights
from common import convert_complex_sh_to_real, load_vsh_coefficients, load_all_vsh_coefficients, make_l_and_m_lists, read_input_NMPostProcess, read_eigenvalues

def read_mode_info(dir_NM, option, i_clip = None):

    # Define directories and files.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Load the mode identification information.
    path_ids = os.path.join(dir_processed, 'mode_ids_{:}.txt'.format(option))
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T
    if i_clip is not None:

        i_mode = i_mode[:i_clip]
        l = l[:i_clip]
        type_ = type_[:i_clip]
        shell = shell[:i_clip]

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)

    ## Read coefficients.
    #coeffs = load_all_vsh_coefficients(dir_NM, i_mode)

    if i_clip is not None:

        f = f[:i_clip]

    return i_mode, f, l, type_, shell 

def coeff_match(coeffs_1, coeffs_2):

    coeffs_1_flat = coeffs_1.flatten()
    coeffs_2_flat = coeffs_2.flatten()

    coeffs_1_norm = np.sqrt(np.sum(coeffs_1_flat**2.0))
    coeffs_2_norm = np.sqrt(np.sum(coeffs_2_flat**2.0))

    coeffs_dot = np.sum(coeffs_1_flat*coeffs_2_flat)
    coeffs_dot_normalised = coeffs_dot/(coeffs_1_norm*coeffs_2_norm)

    return coeffs_dot_normalised

def main():
    
    print('compare.py:')
    #i_clip = 10
    i_clip = None

    # Read the input file and data of the first run.
    dir_PM_1, dir_NM_1, option, l_max_1, _, _ = \
        read_input_NMPostProcess()
    #i_mode_1, f_1, l_1, type_1, shell_1, coeffs_cplx_1 = \
    i_mode_1, f_1, l_1, type_1, shell_1 = \
        read_mode_info(dir_NM_1, option, i_clip = i_clip)
    dir_processed = os.path.join(dir_NM_1, 'processed')

    # Read the data of the second run.
    file_compare = 'input_compare.txt'
    with open(file_compare, 'r') as in_id:

        dir_NM_2 = in_id.readline().strip()

    # Read the data of the second run.
    #i_mode_2, f_2, l_2, type_2, shell_2, coeffs_cplx_2 = \
    i_mode_2, f_2, l_2, type_2, shell_2 = \
        read_mode_info(dir_NM_2, option, i_clip = i_clip)

    path_fit = os.path.join(dir_processed, 'comparison_fit_{:}.npy'.format(option))
    path_index_list = os.path.join(dir_processed, 'comparison_fit_indices_{:}.npy'.format(option))
    if os.path.exists(path_fit):

        print('Fit information already exists, loading: {:}'.format(path_fit))
        fit_list = np.load(path_fit, allow_pickle = True)
        index_list = np.load(path_index_list, allow_pickle = True)
        n_mode_1 = len(fit_list)

    else:

        # Read coefficients.
        print('Reading coefficients.')
        coeffs_cplx_1 = load_all_vsh_coefficients(dir_NM_1, i_mode_1, option = option)
        coeffs_cplx_2 = load_all_vsh_coefficients(dir_NM_2, i_mode_2, option = option)

        # Apply weight information (full mode only).
        if option == 'full':

            # Read the shell radii.
            _, header_info, _, _ = load_vsh_coefficients(dir_NM_1, i_mode_1[0], i_radius = 0)

            # Calculate weights of each shell.
            volume_weights = calculate_weights(header_info['r_sample'])

            # Apply the weights.
            coeffs_cplx_1 = coeffs_cplx_1*volume_weights[:, np.newaxis, np.newaxis]
            coeffs_cplx_2 = coeffs_cplx_2*volume_weights[:, np.newaxis, np.newaxis]

        # Add empty dimension so the shape is the same for quick and full runs.
        if option == 'quick':
            
            coeffs_cplx_1 = np.expand_dims(coeffs_cplx_1, 1)
            coeffs_cplx_2 = np.expand_dims(coeffs_cplx_2, 1)

        # Count the number of radii.
        n_radii = coeffs_cplx_1.shape[1]

        # Scale the coefficients by the k values.
        # First, get the list of l and m values of the coefficients.
        l_max = l_max_1
        l, m = make_l_and_m_lists(l_max)
        k = np.sqrt(l*(l + 1.0))
        coeffs_cplx_1 = coeffs_cplx_1*k
        coeffs_cplx_2 = coeffs_cplx_2*k

        # Convert the coefficients from complex to real.
        n_mode_1 = coeffs_cplx_1.shape[0]
        n_mode_2 = coeffs_cplx_2.shape[0]
        n_real_coeffs = (l_max + 1)*(l_max + 1)
        coeffs_real_1 = np.zeros((n_mode_1, n_radii, 3, n_real_coeffs))
        coeffs_real_2 = np.zeros((n_mode_2, n_radii, 3, n_real_coeffs))
        print('Converting complex coefficients to real: First run.')
        for i in range(n_mode_1):

            for j in range(n_radii):

                for k in range(3):
                    
                    coeffs_real_1[i, j, k, :], _, _ = convert_complex_sh_to_real(
                                                coeffs_cplx_1[i, j, k, :], l_max)

        print('Converting complex coefficients to real: Second run.')
        for i in range(n_mode_2):

            for j in range(n_radii):

                for k in range(3):
                    
                    coeffs_real_2[i, j, k, :], _, _ = convert_complex_sh_to_real(
                                                coeffs_cplx_2[i, j, k, :], l_max)

        # Find the best match for each mode in the first list by comparison
        # with the coefficients in the second list.
        f_thresh = 0.15 # mHz.
        best_match = np.zeros(n_mode_1, dtype = np.int)
        fit_list = []
        index_list = []
        for i in range(n_mode_1):

            print('Calculating fit for mode {:>5d}'.format(i))

            # Narrow down the modes for comparison by using the frequency
            # threshold.
            j_thresh_match = np.where((np.abs(f_1[i] - f_2) < f_thresh))[0]

            # For each mode matching the frequency, compare the coefficients.
            n_thresh_match = len(j_thresh_match)
            fit = np.zeros(n_thresh_match)

            for k, j in enumerate(j_thresh_match):

                fit[k] = coeff_match(coeffs_real_1[i, ...], coeffs_real_2[j, ...])

            #fit_list_array = np.array([j_thresh_match, fit], dtype = [('indices', np.int), ('fit', np.float)])

            index_list.append(j_thresh_match)
            fit_list.append(fit)
            best_match[i] = j_thresh_match[np.argmax(np.abs(fit))]

        # Save fit.
        print('Saving fit information to {:} and {:}.'.format(path_fit, path_index_list))
        np.save(path_fit, fit_list)
        np.save(path_index_list, index_list)

    print('Calculating best match.')
    min_fit = 0.75
    best_match = np.zeros(n_mode_1, dtype = np.int)
    best_fit = np.zeros(n_mode_1)
    for i in range(n_mode_1):

        # Get indices of modes matching frequency threshold.
        j_thresh_match = index_list[i]
        
        # Get fit for this mode.
        fit = fit_list[i]
        abs_fit = np.abs(fit)
        argmax_abs_fit = np.argmax(abs_fit)
        max_abs_fit = abs_fit[argmax_abs_fit]

        best_fit[i] = max_abs_fit

        if max_abs_fit > min_fit:

            # Find the best match.
            best_match[i] = j_thresh_match[argmax_abs_fit]#np.argmax(np.abs(fit))]

        else:

            best_match[i] = -1
        
    print('Best-matching modes:')
    for i in range(n_mode_1):

        if best_match[i] == -1:

            #print('{:>5d} No good matches.'.format(i))
            pass

        else:

            print('{:>5d} {:>5d}, {:5.1f} %'.format(i_mode_1[i], i_mode_2[best_match[i]], 100.0*best_fit[i]))

    path_out = os.path.join(dir_processed, 'comparison.txt')
    print('Saving match information to {:}'.format(path_out))
    with open(path_out, 'w') as out_id:
        
        for i in range(n_mode_1):

            out_id.write('{:>5d} {:>5d}\n'.format(i, best_match[i]))

    #best_fit = np.zeros(n_mode_1)
    #for i in range(n_mode_1):

    ##    best_fit[i] = np.abs(fit_list[i][best_match[i]])
    #    best_fit[i] = np.max(np.abs(fit_list[i]))

    best_match_sorted = np.sort(best_match)
    matched, n_match = np.unique(best_match_sorted, return_index = False, return_inverse = False, return_counts = True)
    if best_match_sorted[0] == -1:
        n_unmatched = n_match[0]
        matched = matched[1:]
        n_match = n_match[1:]

    i_multiply_counted = np.where(n_match > 1)[0]

    for i in i_multiply_counted:

        match = matched[i]
        j_match = np.where(best_match == match)[0]

        print('\n')
        print('The following modes in the first list all match mode {:>5d} in the second list'.format(i_mode_2[match]))
        for j in j_match:
            print('{:>5d}'.format(i_mode_1[j]))

    print('Number of modes not matched: {:>5d}'.format(n_unmatched))
    print('Number of multiply-counted modes: {:>5d}'.format(np.sum(n_match - 1)))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax  = plt.gca()

    dir_processed = os.path.join(dir_NM_1, 'processed')
    dir_plot = os.path.join(dir_processed, 'plots')
    path_fig = os.path.join(dir_plot, 'comparison_mode_diagram_{:}.png'.format(option))

    match_condition = (best_match > -1)
    i_matched = np.where(match_condition)[0]
    i_not_matched = np.where(~match_condition)[0]

    f_diff = f_1 - f_2[best_match]
    abs_f_diff = np.abs(f_diff)
    print(np.max(abs_f_diff[i_matched]))
    print(np.max(f_1))
    ax.scatter(l_1[i_not_matched], f_1[i_not_matched], c = 'r', alpha = 1.0, s = 3)
    ax.scatter(l_1[i_matched], f_1[i_matched], c = 'k', alpha = 1.0, s = 1.0E4*abs_f_diff[i_matched])

    ax.scatter([], [], c = 'r', s = 3, label = 'Not matched')
    ax.scatter([], [], c = 'k', s = 10, label = 'Matched (0.01 $\mu$Hz)')
    ax.scatter([], [], c = 'k', s = 100, label = 'Matched (0.10 $\mu$Hz)')

    font_size_label = 14
    ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
    ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)

    ax.legend()

    plt.tight_layout()

    print('Saving to {:}'.format(path_fig))
    plt.savefig(path_fig, dpi = 300)

    plt.show()

    #fig = plt.figure()
    #ax = plt.gca()

    #dir_processed = os.path.join(dir_NM_1, 'processed')
    #dir_plot = os.path.join(dir_processed, 'plots')
    #path_fig = os.path.join(dir_plot, 'comparison_similarity_{:}.png'.format(option))

    #ax.hist(best_fit)

    #font_size_label = 14
    #ax.set_xlabel('Similarity', fontsize = font_size_label)
    #ax.set_ylabel('Number of modes', fontsize = font_size_label)

    #ax.axvline(min_fit, linestyle = ':', color = 'k')

    #plt.tight_layout()

    #print('Saving to {:}'.format(path_fig))
    #plt.savefig(path_fig, dpi = 300)


    #plt.show()

    return

if __name__ == '__main__':

    main()
