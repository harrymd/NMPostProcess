'''
NMPostProcess/characterise.py
Functions for characterising and identifying modes based on their vector spherical harmonic expansions.
'''

# Import modules. -------------------------------------------------------------

# Import standard modules.
import argparse
import os

# Import third-part modules.
import numpy as np

# Import local modules.
from NMPostProcess.common import convert_complex_sh_to_real, get_list_of_modes_from_coeff_files, load_vsh_coefficients, make_l_and_m_lists, read_input_NMPostProcess
from NMPostProcess.project_into_mode_basis import get_major_modes, load_1d_mode_info

# Characterisation of modes based on their VSH coefficients. ------------------
def calculate_power_distribution_quick_real(Ulm, Vlm, Wlm, l_list, print_ = True):
    '''
    Calculates the 'power' (square of spherical harmonic coefficients) of a surface vector field in the radial, poloidal, and toroidal components, and as a function of the angular order l.

    Input:

    Ulm, Vlm, Wlm, l_list
        See 'Definitions of variables' in NMPostProcess/process.py.
    print_
        If True, print detailed information.

    Output:

    EU, EV, EW, E, E_of_l
        See 'Definitions of variables' in NMPostProcess/process.py.
    '''

    # Calculate 'power' in each component (radial, consoidal, toroidal) and total 'power'.
    EU = np.sum(np.abs(Ulm)**2.0)
    EV = np.sum(np.abs(Vlm)**2.0)
    EW = np.sum(np.abs(Wlm)**2.0)
    # Normalise.
    E   = EU + EV + EW
    
    # Calculate energy as a function of l.
    l_max   = np.max(l_list)
    E_of_l  = np.zeros((l_max + 1))
    for l in range(l_max + 1):
        
        i = (l_list == l)
        E_of_l[l] = (       np.sum(np.abs(Ulm[i])**2.0)
                        +   np.sum(np.abs(Vlm[i])**2.0)
                        +   np.sum(np.abs(Wlm[i])**2.0))
    
    # Print a summary.
    if print_:
    
        print('Power in components:\n{:5.1f} % spheroidal ({:5.1f} % radial, {:5.1f} % consoidal)\n{:5.1f} % toroidal'.format(((EU + EV)/E)*100.0, (EU/E)*100.0, (EV/E)*100.0, (EW/E)*100.0))
    
        print('Power per angular order:')
        for l in range(l_max + 1):
        
            print('{:8d} {:5.1f} %'.format(l, (E_of_l[l]/E)*100.0))
            
    return EU, EV, EW, E, E_of_l

def calculate_power_distribution_quick_cplx(Ulm_real, Vlm_real, Wlm_real, Ulm_imag, Vlm_imag, Wlm_imag, l_list, print_ = True):
    '''
    Calculates the 'power' (square of spherical harmonic coefficients) of a surface vector field in the radial, poloidal, and toroidal components, and as a function of the angular order l.

    Input:

    Ulm, Vlm, Wlm, l_list
        See 'Definitions of variables' in NMPostProcess/process.py.
    print_
        If True, print detailed information.

    Output:

    EU, EV, EW, E, E_of_l
        See 'Definitions of variables' in NMPostProcess/process.py.
    '''
    
    # Convert from complex representation to real representation.
    l_max = np.max(l_list)
    Ulm_real, l_list, m_list = convert_complex_sh_to_real(Ulm_real, l_max)
    Vlm_real, _, _ = convert_complex_sh_to_real(Vlm_real, l_max)
    Wlm_real, _, _ = convert_complex_sh_to_real(Wlm_real, l_max)
    #
    Ulm_imag, l_list, m_list = convert_complex_sh_to_real(Ulm_imag, l_max)
    Vlm_imag, _, _ = convert_complex_sh_to_real(Vlm_imag, l_max)
    Wlm_imag, _, _ = convert_complex_sh_to_real(Wlm_imag, l_max)

    abs_Ulm = np.abs(Ulm_real + 1.0j*Ulm_imag)
    abs_Vlm = np.abs(Vlm_real + 1.0j*Vlm_imag)
    abs_Wlm = np.abs(Wlm_real + 1.0j*Wlm_imag)

    # Calculate 'power' in each component (radial, consoidal, toroidal) and total 'power'.
    EU = np.sum(abs_Ulm**2.0)
    EV = np.sum(abs_Vlm**2.0)
    EW = np.sum(abs_Wlm**2.0)
    # Normalise.
    E   = EU + EV + EW
    
    # Calculate energy as a function of l.
    E_of_l  = np.zeros((l_max + 1))
    for l in range(l_max + 1):
        
        i = (l_list == l)
        E_of_l[l] = (       np.sum(abs_Ulm[i]**2.0)
                        +   np.sum(abs_Vlm[i]**2.0)
                        +   np.sum(abs_Wlm[i]**2.0))
    
    # Print a summary.
    if print_:
    
        print('Power in components:\n{:5.1f} % spheroidal ({:5.1f} % radial, {:5.1f} % consoidal)\n{:5.1f} % toroidal'.format(((EU + EV)/E)*100.0, (EU/E)*100.0, (EV/E)*100.0, (EW/E)*100.0))
    
        print('Power per angular order:')
        for l in range(l_max + 1):
        
            print('{:8d} {:5.1f} %'.format(l, (E_of_l[l]/E)*100.0))
            
    return EU, EV, EW, E, E_of_l

def calculate_power_distribution_full(coeffs, l_list, weights, print_ = True):
    '''
    Calculates the 'power' (square of spherical harmonic coefficients) of a surface vector field in the radial, poloidal, and toroidal components, and as a function of the angular order l.

    Input:

    Ulm, Vlm, Wlm, l_list
        See 'Definitions of variables' in NMPostProcess/process.py.
    print_
        If True, print detailed information.

    Output:

    EU, EV, EW, E, E_of_l
        See 'Definitions of variables' in NMPostProcess/process.py.
    '''

    # Apply the weights.
    coeffs = coeffs*weights[:, np.newaxis, np.newaxis]

    # Multiply the V and W coefficients by k.
    k = np.sqrt(l_list*(l_list + 1))
    coeffs[:, 1, :] = k*coeffs[:, 1, :]
    coeffs[:, 2, :] = k*coeffs[:, 2, :]

    # Calculate 'power' in each component (radial, consoidal, toroidal) and total 'power'.
    EU = np.sum(np.abs(coeffs[:, 0, :])**2.0)
    EV = np.sum(np.abs(coeffs[:, 1, :])**2.0)
    EW = np.sum(np.abs(coeffs[:, 2, :])**2.0)
    # Normalise.
    E   = EU + EV + EW

    # Calculate energy as a function of l.
    l_max   = np.max(l_list)
    E_of_l  = np.zeros((l_max + 1))
    for l in range(l_max + 1):
        
        i = (l_list == l)
        E_of_l[l] = (       np.sum(np.abs(coeffs[:, 0, i])**2.0)
                        +   np.sum(np.abs(coeffs[:, 1, i])**2.0)
                        +   np.sum(np.abs(coeffs[:, 2, i])**2.0))
    
    # Print a summary.
    if print_:
    
        print('Power in components:\n{:5.1f} % spheroidal ({:5.1f} % radial, {:5.1f} % consoidal)\n{:5.1f} % toroidal'.format(((EU + EV)/E)*100.0, (EU/E)*100.0, (EV/E)*100.0, (EW/E)*100.0))
    
        print('Power per angular order:')
        for l in range(l_max + 1):
        
            print('{:8d} {:5.1f} %'.format(l, (E_of_l[l]/E)*100.0))
            
    return EU, EV, EW, E, E_of_l

def characterise_all_modes_quick(dir_NM): 

    option = 'quick'
    i_radius = None

    # Get a list of modes.
    #i_mode_list = get_list_of_modes_from_output_files(dir_NM)
    i_mode_list = get_list_of_modes_from_coeff_files(dir_NM, option) 

    n_modes = len(i_mode_list)

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Loop over all modes.
    first_iteration = True
    for i, i_mode in enumerate(i_mode_list):

        print('Characterising mode {:>5d}'.format(i_mode))

        coeffs, header_info, r_sample, i_sample = \
            load_vsh_coefficients(dir_NM, i_mode, i_radius = i_radius)

        if coeffs.shape[0] == 3:

            modes_are_complex = False
            Ulm, Vlm, Wlm = coeffs

        elif coeffs.shape[0] == 6:

            modes_are_complex = True
            Ulm_real, Vlm_real, Wlm_real, Ulm_imag, Vlm_imag, Wlm_imag = coeffs

        else:

            raise ValueError

        if first_iteration:

            # Infer the maximum l-value used.
            if modes_are_complex:

                n_coeffs = len(Ulm_real)

            else:

                n_coeffs = len(Ulm)

            l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

            # Get the list of l and m values of the coefficients.
            l, m = make_l_and_m_lists(l_max)

            # Prepare output arrays.
            E_UVW = np.zeros((3, n_modes))
            E_of_l = np.zeros((l_max + 1, n_modes))
            r_max = np.zeros(n_modes)
            i_region_max = np.zeros(n_modes, dtype = np.int)

            first_iteration = False

        # Calculate the power distribution.
        if modes_are_complex:

            EU_i, EV_i, EW_i, E_i, E_of_l_i = \
                calculate_power_distribution_quick_cplx(
                    Ulm_real, Vlm_real, Wlm_real,
                    Ulm_imag, Vlm_imag, Wlm_imag,
                    l, print_ = False)

        else:

            EU_i, EV_i, EW_i, E_i, E_of_l_i = calculate_power_distribution_quick_real(Ulm, Vlm, Wlm, l, print_ = False)

        # Normalise by the sum.
        EU_i        = EU_i/E_i
        EV_i        = EV_i/E_i
        EW_i        = EW_i/E_i
        E_of_l_i    = E_of_l_i/E_i

        # Store.
        E_UVW[0, i]     = EU_i
        E_UVW[1, i]     = EV_i
        E_UVW[2, i]     = EW_i
        E_of_l[:, i]    = E_of_l_i
        # 
        r_max[i] = r_sample
        i_region_max[i] = i_sample

    # Save mode characterisation information.
    array_out = np.array([*E_UVW, *E_of_l, r_max, i_region_max])
    path_out = os.path.join(dir_NM, 'processed', 'characterisation_quick.npy')
    print('Saving mode characterisation information to {:}'.format(path_out))
    np.save(path_out, array_out)

    # Find l-value and type for each mode.
    l = np.zeros(n_modes, dtype = np.int)
    shell = np.zeros(n_modes, dtype = np.int)
    # type_: 0: radial, 1: spheroidal, 2: toroidal
    type_ = np.zeros(n_modes, dtype = np.int)
    for i in range(n_modes):

        shell[i] = i_region_max[i]//3
         
        # Find dominant l-value.
        l[i] = np.argmax(E_of_l[:, i])

        if l[i] == 0:

            type_[i] = 0

        else:

            # Check if spheroidal or toroidal power is greater.
            if ((E_UVW[0, i] + E_UVW[1, i]) > E_UVW[2, i]):

                type_[i] = 1

            else:

                type_[i] = 2

    path_out_ids = os.path.join(dir_NM, 'processed', 'mode_ids_quick.txt')
    out_array_ids = np.array([i_mode_list, l, type_, shell])
    print('Saving mode identifications to {:}'.format(path_out_ids))
    np.savetxt(path_out_ids, out_array_ids.T, fmt = '%i')

    return

def calculate_weights(r_sample):
    '''
    Calculates weights proportional to volume of spherical shell closest to
    each sample.
    '''

    n_sample = len(r_sample)
    weights = np.zeros(n_sample)

    for i in range(n_sample):

        if i == 0:

            r_outer = r_sample[0]
            r_inner = 0.5*(r_sample[1] + r_sample[0])

        elif i == (n_sample - 1):

            r_outer = 0.5*(r_sample[i - 1] + r_sample[i])
            r_inner = 0.0

        else:

            r_outer = 0.5*(r_sample[i - 1] + r_sample[i])
            r_inner = 0.5*(r_sample[i] + r_sample[i + 1])

        # Proportinal to volume.
        weights[i] = (r_outer**3.0) - (r_inner**3.0)

    # Normalise the weights.
    weights_sum = np.sum(weights) 
    weights = weights/weights_sum
    
    return weights

def characterise_all_modes_full(dir_NM): 

    option = 'full'
    #i_radius = None

    # Get a list of modes.
    #i_mode_list = get_list_of_modes_from_output_files(dir_NM)
    i_mode_list = get_list_of_modes_from_coeff_files(dir_NM, option) 
    n_modes = len(i_mode_list)

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Loop over all modes.
    first_iteration = True
    for i, i_mode in enumerate(i_mode_list):
        
        print('Characterising mode {:>5d}, item {:>5d} of {:>5d}'.format(i_mode, i + 1, n_modes))

        # Load the complex VSH coefficients.
        coeffs, header_info, _, _ = load_vsh_coefficients(dir_NM, i_mode, i_radius = 'all')
        
        if coeffs.shape[1] == 3:

            modes_are_complex = False

        elif coeffs.shape[1] == 6:

            modes_are_complex = True
            coeffs_new = np.zeros((coeffs.shape[0], 3, coeffs.shape[2]),
                            dtype = np.complex)
            coeffs_new[:, 0, :] = coeffs[:, 0, :] + 1.0j*coeffs[:, 3, :]
            coeffs_new[:, 1, :] = coeffs[:, 1, :] + 1.0j*coeffs[:, 4, :]
            coeffs_new[:, 2, :] = coeffs[:, 2, :] + 1.0j*coeffs[:, 5, :]

            coeffs = coeffs_new

        if first_iteration:

            # Infer the maximum l-value used.
            n_coeffs = coeffs.shape[2] 
            l_max = (int((np.round(np.sqrt(8*n_coeffs + 1)) - 1))//2) - 1

            # Get the list of l and m values of the coefficients.
            l, m = make_l_and_m_lists(l_max)

            # Prepare output arrays.
            E_UVW = np.zeros((3, n_modes))
            E_of_l = np.zeros((l_max + 1, n_modes))
            r_max = np.zeros(n_modes)
            i_region_max = np.zeros(n_modes, dtype = np.int)

            # Calculate weights of each shell.
            volume_weights = calculate_weights(header_info['r_sample'])

            first_iteration = False

        # Calculate the power distribution.
        EU_i, EV_i, EW_i, E_i, E_of_l_i = calculate_power_distribution_full(coeffs, l, volume_weights, print_ = False)

        # Normalise by the sum.
        EU_i        = EU_i/E_i
        EV_i        = EV_i/E_i
        EW_i        = EW_i/E_i
        E_of_l_i    = E_of_l_i/E_i

        # Store.
        E_UVW[0, i]     = EU_i
        E_UVW[1, i]     = EV_i
        E_UVW[2, i]     = EW_i
        E_of_l[:, i]    = E_of_l_i
        # 
        r_max[i] = header_info['r_max']
        i_region_max[i] = header_info['i_region_max']

    # Save mode characterisation information.
    array_out = np.array([*E_UVW, *E_of_l, r_max, i_region_max])
    path_out = os.path.join(dir_NM, 'processed', 'characterisation_full.npy')
    print('Saving mode characterisation information to {:}'.format(path_out))
    np.save(path_out, array_out)

    # Find l-value and type for each mode.
    l = np.zeros(n_modes, dtype = np.int)
    shell = np.zeros(n_modes, dtype = np.int)
    # type_: 0: radial, 1: spheroidal, 2: toroidal
    type_ = np.zeros(n_modes, dtype = np.int)
    for i in range(n_modes):

        shell[i] = i_region_max[i]//3
         
        # Find dominant l-value.
        l[i] = np.argmax(E_of_l[:, i])

        if l[i] == 0:

            type_[i] = 0

        else:

            # Check if spheroidal or toroidal power is greater.
            if ((E_UVW[0, i] + E_UVW[1, i]) > E_UVW[2, i]):

                type_[i] = 1

            else:

                type_[i] = 2

    path_out_ids = os.path.join(dir_NM, 'processed', 'mode_ids_full.txt')
    out_array_ids = np.array([i_mode_list, l, type_, shell])
    print('Saving mode identifications to {:}'.format(path_out_ids))
    np.savetxt(path_out_ids, out_array_ids.T, fmt = '%i')

    return

def characterise_all_modes_projection(dir_NM, path_1d_input):

    # Get 1-D mode information.
    mode_info_1d, model_1d, run_info_1d = load_1d_mode_info(path_1d_input)

    # Get a list of modes.
    option = 'full'
    i_mode_list = get_list_of_modes_from_coeff_files(dir_NM, option) 
    num_modes = len(i_mode_list)

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_projections = os.path.join(dir_processed, 'projections')

    # Find l-value and type for each mode.
    n = np.zeros(num_modes, dtype = np.int)
    l = np.zeros(num_modes, dtype = np.int)
    type_ = np.zeros(num_modes, dtype = np.int)
    amp_thresh = 0.2
    type_str_to_int_dict = {'R' : 0, 'S' : 1, 'T0' : 2, 'T1' : 3}
    for i in range(num_modes):

        print('Mode: {:>5d}'.format(i_mode_list[i]))

        n_i, l_i, f_i, amp_i, type_str_i = get_major_modes(
                dir_projections, mode_info_1d, i_mode_list[i], amp_thresh)

        n[i] = n_i[0]
        l[i] = l_i[0]
        #f[i] = f_i[0]
        type_[i] = type_str_to_int_dict[type_str_i[0]]

    path_out_ids = os.path.join(dir_NM, 'processed', 'mode_ids_projection.txt')
    out_array_ids = np.array([i_mode_list, n, l, type_])
    print('Saving mode identifications to {:}'.format(path_out_ids))
    np.savetxt(path_out_ids, out_array_ids.T, fmt = '%i')

    return

# Main. -----------------------------------------------------------------------
def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_projection", metavar = 'path_input_1d') 
    args = parser.parse_args()
    path_1d_input  = args.from_projection
    
    # Read input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii  = read_input_NMPostProcess()

    if path_1d_input is not None:

        characterise_all_modes_projection(dir_NM, path_1d_input)
        
    elif option == 'quick':

        characterise_all_modes_quick(dir_NM)

    elif option == 'full':

        characterise_all_modes_full(dir_NM)

    else:

        raise ValueError('Option {:} not recognised.'.format(option))

    return

if __name__ == '__main__':

    main()
