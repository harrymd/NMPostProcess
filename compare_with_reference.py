import os

import numpy as np

from stoneley.code.Ouroboros.common import get_Ouroboros_out_dirs, read_Ouroboros_input_file
from common import mode_id_information_to_dict, read_eigenvalues, read_input_NMPostProcess, reference_mode_info_to_dict

def main():

    # Read the NMPostProcess input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii = read_input_NMPostProcess()

    # Load the mode identification information.
    dir_processed = os.path.join(dir_NM, 'processed')
    path_ids = os.path.join(dir_processed, 'mode_ids_{:}.txt'.format(option))
    i_mode, l, type_, shell = np.loadtxt(path_ids, dtype = np.int).T

    # Read the mode frequency. 
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f = read_eigenvalues(file_eigval_list)
    num_modes = len(f)

    # Read 3-D mode information.
    mode_info = mode_id_information_to_dict(type_, l, f, shell)

    # Find 1-D mode information.
    Ouroboros_input_file = '../Ouroboros/input_Magrathea.txt'
    Ouroboros_info = read_Ouroboros_input_file(Ouroboros_input_file)

    mode_types = list(mode_info.keys())
    NormalModes_to_Ouroboros_T_layer_num_dict = {0 : 1, 2 : 0}
    paths_ref = dict()
    for mode_type in mode_types:
        
        if mode_type[0] == 'T':

            _, _, _, dir_type = get_Ouroboros_out_dirs(Ouroboros_info, 'T')
            layer_number_NormalModes = int(mode_type[1])
            layer_number_Ouroboros = \
                NormalModes_to_Ouroboros_T_layer_num_dict[layer_number_NormalModes]

            path_ref = os.path.join(dir_type, 'eigenvalues_{:>03d}.txt'.format(layer_number_Ouroboros))

        else:

            _, _, _, dir_type = get_Ouroboros_out_dirs(Ouroboros_info, mode_type)

            path_ref = os.path.join(dir_type, 'eigenvalues.txt')
        
        paths_ref[mode_type] = path_ref

    # Read 1-D reference information.
    nlf_ref = reference_mode_info_to_dict(paths_ref)

    # For each mode, find best fitting reference mode.
    n       = np.zeros(num_modes, dtype = np.int)
    f_ref   = np.zeros(num_modes)
    ref_key_list = []
    for i in range(num_modes):


        if type_[i] == 0:

            ref_key = 'R'

        elif type_[i] == 1:

            ref_key = 'S'

        elif type_[i] == 2:

            ref_key = 'T{:>1d}'.format(shell[i])

        else:

            raise ValueError

        ref_key_list.append(ref_key)

        j_match = np.where(nlf_ref[ref_key]['l'] == l[i])[0]
        k_match = np.argmin(np.abs(nlf_ref[ref_key]['f'][j_match] - f[i]))
        i_match = j_match[k_match]

        n[i] = nlf_ref[ref_key]['n'][i_match]
        f_ref[i] = nlf_ref[ref_key]['f'][i_match]

    f_diff = f - f_ref

    format_string = '{:>5d} {:>1d} {:>4} {:>1d} {:>9.6f} {:>9.6f} {:>+10.6f}'
    print('{:>5} {:>1} {:>4} {:>1} {:>9} {:>9}'.format('Mode', 'n', 'type', 'l', 'f', 'f_ref', 'f_diff'))
    for i in range(num_modes):

        print(format_string.format(i_mode[i], n[i], ref_key_list[i], l[i], f[i], f_ref[i], f_diff[i]))

    return

if __name__ == '__main__':

    main()
