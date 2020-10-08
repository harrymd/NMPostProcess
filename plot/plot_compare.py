import os

import matplotlib.pyplot as plt
import numpy as np

from common import read_eigenvalues, read_input_NMPostProcess

def main():

    # Read the NMPostProcess input file.
    _, dir_NM_1, _, _, _, _ = read_input_NMPostProcess()



    # Read the comparison input file.
    file_compare = 'input_compare.txt'
    with open(file_compare, 'r') as in_id:

        dir_NM_2 = in_id.readline().strip()

    # Read the mode frequencies. 
    dir_processed_1 = os.path.join(dir_NM_1, 'processed')
    file_eigval_list_1        = os.path.join(dir_processed_1, 'eigenvalue_list.txt')
    _, f_1 = read_eigenvalues(file_eigval_list_1)
    #
    dir_processed_2 = os.path.join(dir_NM_2, 'processed')
    file_eigval_list_2        = os.path.join(dir_processed_2, 'eigenvalue_list.txt')
    _, f_2 = read_eigenvalues(file_eigval_list_2)

    # Read the comparison information.
    path_compare = os.path.join(dir_processed_1, 'comparison.txt')
    i_mode_1, i_mode_2 = np.loadtxt(path_compare, dtype = np.int).T

    # Load the mode identification information.
    path_ids_1 = os.path.join(dir_processed_1, 'mode_ids_quick.txt')
    _, l_1, _, _ = np.loadtxt(path_ids_1, dtype = np.int).T
    #i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Sort the frequencies of the comparison run to their best match in the
    # original run.
    f_2_reordered = f_2[i_mode_2]

    # Find minimum and maximum frequencies.
    f_min = np.min(np.concatenate([f_1, f_2]))
    f_max = np.max(np.concatenate([f_1, f_2]))

    # Find maximum frequency difference.
    f_diff = f_1 - f_2_reordered
    f_diff_frac = f_diff/(0.5*(f_1 + f_2_reordered))
    f_diff_max = np.max(np.abs(f_diff))
    print('Maximum frequency difference: {:>12.6} muHz'.format(f_diff_max*1.0E3))

    #fig = plt.figure(figsize = (7.0, 7.0)) 
    #ax  = plt.gca()

    #ax.scatter(f_1, f_2_reordered)

    ## Plot guide line.
    #f_line_buff = 0.2
    #f_min_line = f_min*(1.0 - f_line_buff)
    #f_max_line = f_max*(1.0 + f_line_buff)
    #ax.plot([f_min_line, f_max_line], [f_min_line, f_max_line])

    ## Set axis limits.
    #f_lim_buff = 0.1
    #f_min_lim = f_min*(1.0 - f_lim_buff)
    #f_max_lim = f_max*(1.0 + f_lim_buff)
    #ax.set_xlim([f_min_lim, f_max_lim])
    #ax.set_ylim([f_min_lim, f_max_lim])

    #font_size_label = 12
    #ax.set_xlabel('Frequency (mHz) of mode in run 1', fontsize = font_size_label)
    #ax.set_ylabel('Frequency (mHz) of mode in run 2', fontsize = font_size_label)

    #ax.set_aspect(1.0)

    #plt.show()

    #fig = plt.figure()
    #ax = plt.gca()

    ##ax.plot(i_mode_1, f_diff*1.0E3, 'k.-')
    #ax.plot(i_mode_1, f_diff_frac*1.0E2, 'k.-')

    #font_size_label = 12
    #ax.set_xlabel('Mode number', fontsize = font_size_label)
    ##ax.set_ylabel('Frequency difference ($\mu$Hz)', fontsize = font_size_label)
    #ax.set_ylabel('Frequency difference (%)', fontsize = font_size_label)

    #fig = plt.figure()
    #ax = plt.gca()

    #ax.scatter(f_1, f_diff*1.0E3, c = 'k', alpha = 0.5)
    ##ax.plot(i_mode_1, f_diff_frac*1.0E2, 'k.-')

    #font_size_label = 12
    #ax.set_xlabel('Frequency', fontsize = font_size_label)
    #ax.set_ylabel('Frequency difference ($\mu$Hz)', fontsize = font_size_label)
    ##ax.set_ylabel('Frequency difference (%)', fontsize = font_size_label)

    fig = plt.figure()
    ax = plt.gca()
    
    print(np.max(np.abs(f_diff)))
    
    ax.scatter(l_1, f_1, s = 10*1.0E3*np.abs(f_diff), c = 'k', alpha = 0.5)
    #ax.plot(i_mode_1, f_diff_frac*1.0E2, 'k.-')

    ax.scatter([], [], c = 'k', s = 10.0, label = '1 $\mu Hz$')
    ax.scatter([], [], c = 'k', s = 100.0, label = '10 $\mu Hz$')
    ax.legend()

    font_size_label = 12
    ax.set_xlabel('Angular order, $\ell$', fontsize = font_size_label)
    ax.set_ylabel('Frequency (mHz)', fontsize = font_size_label)
    #ax.set_ylabel('Frequency difference ($\mu$Hz)', fontsize = font_size_label)
    ##ax.set_ylabel('Frequency difference (%)', fontsize = font_size_label)

    plt.show()

if __name__ == '__main__':

    main()
