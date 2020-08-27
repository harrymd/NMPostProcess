'''
NMPostProcess/characterise.py
Functions for characterising and identifying modes based on their vector spherical harmonic expansions.
'''

# Import modules. -------------------------------------------------------------

# Import standard modules.
import os

# Import third-part modules.
import numpy as np

# Import local modules.
from common import get_list_of_modes_from_output_files, load_vsh_coefficients, make_l_and_m_lists

# Characterisation of modes based on their VSH coefficients. ------------------
def calculate_power_distribution(Ulm, Vlm, Wlm, l_list, print_ = True):
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

def characterise_all_modes_quick(dir_NM): 

    # Get a list of modes.
    i_mode_list = get_list_of_modes_from_output_files(dir_NM)
    n_modes = len(i_mode_list)

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')
    
    # Loop over all modes.
    first_iteration = True
    for i, i_mode in enumerate(i_mode_list):

        # Load the complex VSH coefficients.
        Ulm, Vlm, Wlm, scale, r_max_i, i_region_max_i = load_vsh_coefficients(dir_NM, i_mode)

        if first_iteration:

            # Infer the maximum l-value used.
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
        EU_i, EV_i, EW_i, E_i, E_of_l_i = calculate_power_distribution(Ulm, Vlm, Wlm, l, print_ = False)
        
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
        r_max[i] = r_max_i
        i_region_max[i] = i_region_max_i

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

# Main. -----------------------------------------------------------------------
def main():
    
    # Read the input file.
    input_file = 'input_NMPostProcess.txt'
    with open(input_file, 'r') as in_id:

        input_args = in_id.readlines()
    
    # Parse input arguments.
    # Remove trailing newline characters.
    input_args = [x.strip() for x in input_args]
    #
    dir_PM      = input_args[0]
    dir_NM      = input_args[1]
    option      = input_args[2]
    l_max       = int(input_args[3])
    i_mode_str  = input_args[4]

    if option == 'quick':

        characterise_all_modes_quick(dir_NM)

    else:

        raise ValueError('Option {:} not recognised.'.format(option))

if __name__ == '__main__':

    main()
