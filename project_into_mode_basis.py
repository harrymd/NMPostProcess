import argparse
import os

import numpy as np

from NMPostProcess.common import (convert_complex_sh_to_real,
                            load_vsh_coefficients, get_list_of_modes_from_coeff_files,
                            read_eigenvalues, read_input_NMPostProcess)
from Ouroboros.common import (get_n_solid_layers, load_eigenfreq,
                            load_eigenfunc, load_model, read_input_file)

def load_1d_mode_info(path_1d_input):
    
    # Read 1-D input file.
    run_info = read_input_file(path_1d_input)

    # Load the planetary model for 1-D modes.
    model = load_model(run_info['path_model'])

    # Get mode informoation for all mode types.
    mode_info = dict()
    if run_info['code'] == 'mineos':
        
        mode_types = ['R', 'S', 'T', 'I']
        for mode_type in mode_types:

            mode_info[mode_type] = load_eigenfreq(run_info, mode_type)

    elif run_info['code'] == 'ouroboros':

        n_solid_layers = get_n_solid_layers(model)
        
        mode_types = ['R', 'S', 'T']
        for mode_type in mode_types:

            if mode_type in ['R', 'S']:

                mode_info[mode_type] = load_eigenfreq(run_info, mode_type)

            elif mode_type == 'T':

                for i in range(n_solid_layers):

                    mode_type_str = 'T{:>d}'.format(i)
                    mode_info[mode_type_str] = load_eigenfreq(run_info, mode_type,
                                                i_toroidal = i)

    else:

        raise NotImplementedError

    return mode_info, model, run_info

def projection_wrapper_one_mode(dir_NM, i_mode, f, mode_info_1d, run_info_1d, l_max, shell_mass, max_f_diff = 0.2):

    # Find 1-D modes which are close in frequency to the 3-D mode.
    mode_info_1d = find_close_modes(f, mode_info_1d, max_f_diff)

    # Load the eigfunction of the 3-D mode.
    coeffs, header, _, _ = load_vsh_coefficients(dir_NM, i_mode, i_radius = 'all')
    r = header['r_sample']
    n_radii = len(r)

    # Convert to real spherical harmonics.
    l_coeffs, m_coeffs, coeffs = convert_to_real_spherical_harmonics(coeffs, l_max)

    # Multiply V and W components by k.
    k = np.sqrt(l_coeffs * (l_coeffs + 1))
    k[k == 0.0] = 1.0
    coeffs[1, :, :] = k * coeffs[1, :, :]
    coeffs[2, :, :] = k * coeffs[2, :, :]
    coeffs[4, :, :] = k * coeffs[4, :, :]
    coeffs[5, :, :] = k * coeffs[5, :, :]
    
    mode_type_test = 'T1'
    mode_info_1d = {mode_type_test : mode_info_1d[mode_type_test]}

    # Loop over all the 1-D modes and evaluate scalar product with the 3-D mode.
    num_modes_compare = sum([len(mode_info_1d[mode_type]['n']) for mode_type in mode_info_1d.keys()])
    #
    n = np.zeros(num_modes_compare)
    l = np.zeros(num_modes_compare)
    m = np.zeros(num_mode
    #
    product_real_part = np.zeros(num_modes_compare)
    product_imag_part = np.zeros(num_modes_compare)
    i = 0
    for mode_type_1d in mode_info_1d.keys():

        num_modes_type = len(mode_info_1d[mode_type_1d]['n'])
        for j in range(num_modes_type):

            n

            product_real_part[i] = project_one_mode_onto_one_multiplet(
                                    header['r_sample'], header['i_sample'],
                                    shell_mass,
                                    coeffs[0, :, :], coeffs[1, :, :],
                                    coeffs[2, :, :], run_info_1d, mode_type_1d,
                                    mode_info_1d[mode_type_1d]['n'][j],
                                    mode_info_1d[mode_type_1d]['l'][j],
                                    mode_info_1d[mode_type_1d]['f'][j])

            print(product_real_part[i])

            import sys
            sys.exit()

            i = i + 1

    return

def find_close_modes(f, mode_info_1d, max_f_diff):

    f_min = f - max_f_diff
    f_max = f + max_f_diff

    mode_info_1d_filtered = dict()
    for mode_type in mode_info_1d.keys():
        
        f_1d = mode_info_1d[mode_type]['f']
        i_close = np.where((f_1d > f_min) & (f_1d < f_max))[0]

        mode_info_1d_filtered[mode_type] = dict()

        for var in mode_info_1d[mode_type].keys():

            mode_info_1d_filtered[mode_type][var] = \
                mode_info_1d[mode_type][var][i_close]

    return mode_info_1d_filtered

def convert_to_real_spherical_harmonics(coeffs, l_max):

    n_radii = coeffs.shape[0]

    Ulm_real = coeffs[:, 0, :]
    Vlm_real = coeffs[:, 1, :]
    Wlm_real = coeffs[:, 2, :]
    #
    Ulm_imag = coeffs[:, 3, :]
    Vlm_imag = coeffs[:, 4, :]
    Wlm_imag = coeffs[:, 5, :]
    #
    n_coeffs_real = (l_max + 1)*(l_max + 1)
    #
    ulm_real = np.zeros((n_radii, n_coeffs_real))
    vlm_real = np.zeros((n_radii, n_coeffs_real))
    wlm_real = np.zeros((n_radii, n_coeffs_real))
    #
    ulm_imag = np.zeros((n_radii, n_coeffs_real))
    vlm_imag = np.zeros((n_radii, n_coeffs_real))
    wlm_imag = np.zeros((n_radii, n_coeffs_real))
    #
    for i in range(n_radii):
        
        ulm_real[i, :], l, m = convert_complex_sh_to_real(Ulm_real[i, :], l_max)
        vlm_real[i, :], _, _ = convert_complex_sh_to_real(Vlm_real[i, :], l_max)
        wlm_real[i, :], _, _ = convert_complex_sh_to_real(Wlm_real[i, :], l_max)
        #
        ulm_imag[i, :], _, _ = convert_complex_sh_to_real(Ulm_imag[i, :], l_max)
        vlm_imag[i, :], _, _ = convert_complex_sh_to_real(Vlm_imag[i, :], l_max)
        wlm_imag[i, :], _, _ = convert_complex_sh_to_real(Wlm_imag[i, :], l_max)

    out_arr = np.array([ulm_real, vlm_real, wlm_real, ulm_imag, vlm_imag, wlm_imag])

    return l, m, out_arr

def get_mid_points(i_sample, r_sample):

    n_sample = len(r_sample)
    r_outer = np.zeros(n_sample)
    r_inner = np.zeros(n_sample)

    for i in range(n_sample):

        if (i == 0) or (i_sample[i] in [3, 6]):
            
            r_outer[i] = r_sample[i]
            r_inner[i] = (r_sample[i] + r_sample[i + 1])/2.0

        elif (i == (n_sample - 1)) or (i_sample[i] in [2, 5]):
            
            r_outer[i] = (r_sample[i] + r_sample[i - 1])/2.0
            r_inner[i] = r_sample[i]

        else:

            r_outer[i] = (r_sample[i] + r_sample[i - 1])/2.0
            r_inner[i] = (r_sample[i] + r_sample[i + 1])/2.0

    return r_inner, r_outer

def get_shell_mass(i_sample, r_inner, r_outer, model_1d, normalise = True):

    # Convert from m to km.
    model_1d['r'] = model_1d['r']*1.0E-3

    # Reverse sample lists.
    i_sample = i_sample[::-1]
    r_inner  = r_inner[::-1]
    r_outer  = r_outer[::-1]

    # Get indices of discontinuities in the 1-D model.
    i_outer_core = np.where(model_1d['v_s'] == 0.0)[0]
    i_icb_model = i_outer_core[0]
    i_cmb_model = i_outer_core[-1] + 1

    # Break the density profile into segments and integrate to get mass.
    n_shells = len(i_sample)
    shell_mass = np.zeros(n_shells)
    for j in range(n_shells):

        if i_sample[j] in [0, 1, 2]:
            
            k1 = None 
            k0 = i_cmb_model

        elif i_sample[j] in [3, 4, 5]:

            k1 = i_cmb_model
            k0 = i_icb_model

        elif i_sample[j] in [6, 7]:

            k1 = i_icb_model
            k0 = 0

        else:

            raise ValueError

        r_model   = model_1d['r']  [k0 : k1]
        rho_model = model_1d['rho'][k0 : k1]
        
        p_shell = np.where((r_model <= r_outer[j]) & (r_model >= r_inner[j]))[0]
        #
        r_shell   = r_model[p_shell]
        rho_shell = rho_model[p_shell]
        
        if r_shell[0] != r_inner[j]:
            
            rho_inner_j = np.interp(r_inner[j], r_model, rho_model)
            #
            r_shell     = np.insert(r_shell,   0, r_inner[j])
            rho_shell   = np.insert(rho_shell, 0, rho_inner_j)

        if r_shell[-1] != r_outer[j]:

            rho_outer_j = np.interp(r_outer[j], r_model, rho_model)
            #
            r_shell     = np.insert(r_shell,   len(r_shell),   r_outer[j])
            rho_shell   = np.insert(rho_shell, len(rho_shell), rho_outer_j)

        # Define mass per unit thickness function.
        func = rho_shell * (r_shell ** 2.0)

        # Integrate.
        shell_mass[j] = np.trapz(func, x = r_shell)

    # Normalise.
    if normalise:

        total_mass = np.sum(shell_mass)
        shell_mass = shell_mass/total_mass

    return shell_mass

def get_shell_mass_wrapper(dir_NM, model_1d):
#def load_model_info(dir_base, RadialPNM_info, r_q, r_q_vals):

    # Load radius information for 3-D modes.
    _, header, _, _ = load_vsh_coefficients(dir_NM, 1, i_radius = 'all')

    # Calculate the mid-points.
    r_inner, r_outer = get_mid_points(header['i_sample'], header['r_sample'])

    # Extract the portions of the density model for each shell.
    shell_mass = get_shell_mass(header['i_sample'], r_inner, r_outer, model_1d, normalise = True)

    return shell_mass

def project_one_mode_one_basis(r_sample, i_sample, shell_mass, U_3d, V_3d, W_3d, run_info_1d, type_1d, n_1d, l_1d, f_1d):

    print(U_3d.shape)

    # Mineos and Ouroboros use slightly different mode labelling.
    if run_info_1d['code'] == 'ouroboros':

        if type_1d[0] == 'T':

            i_toroidal = int(type_1d[1])
            type_1d = 'T'

        else:

            i_toroidal = None

    else:

        i_toroidal = None

    # Load the 1-D eigenfunction.
    f_1d_rad_per_s = f_1d*1.0E-3*2.0*np.pi
    norm_func = 'DT'
    norm_units = 'mineos'
    normalisation_args = {'norm_func' : norm_func, 'units' : norm_units}
    normalisation_args['omega'] = f_1d_rad_per_s
    #
    eigfunc_dict = load_eigenfunc(run_info_1d, type_1d, n_1d, l_1d,
                        norm_args = normalisation_args,
                        i_toroidal = i_toroidal)
    eigfunc_dict['r'] = eigfunc_dict['r']*1.0E-3 # Convert to km.

    # Interpolate 1-D eigenfunction at 3-D sample locations.
    eigfunc_interpolated = interpolate_eigfunc(r_sample, i_sample,
                                eigfunc_dict, type_1d, i_toroidal = i_toroidal)

    # Evaluate the scalar product.
    n_radii = len(r_sample)
    product = 0.0
    for i in range(n_radii):
        
        if type_1d == 'R':

            product_i = (shell_mass[i] * U_3d[i] * eigfunc_interpolated['U'][i])

        elif type_1d == 'S': 

            product_i = shell_mass[i] * (U_3d[i] * eigfunc_interpolated['U'][i] +
                                         V_3d[i] * eigfunc_interpolated['V'][i])

        elif type_1d[0] == 'T':

            product_i = (shell_mass[i] * W_3d[i] * eigfunc_interpolated['W'][i])

        product = product + product_i

    print(product)

    return product

def interpolate_eigfunc(r_sample, i_sample, eigfunc_dict, mode_type, i_toroidal = None):
    
    if i_toroidal is None:

        i_discon = np.where(np.diff(eigfunc_dict['r']) == 0.0)[0]
        i_icb = i_discon[0] + 1
        i_cmb = i_discon[1] + 1

    n_radii = len(r_sample)
    eigfunc_interpolated = dict()

    if mode_type in ['R', 'S']:

        if mode_type == 'R':
        
            keys = ['U']

        elif mode_type == 'S':

            keys = ['U', 'V']

        for key in keys:

            eigfunc_interpolated[key] = np.zeros(n_radii)

        # Interpolate the 1-D eigenfunction at the 3-D radial coordinates.
        for j in range(n_radii):

            if i_sample[j] in [0, 1, 2]:

                k0 = i_cmb
                k1 = None 

            elif i_sample[j] in [3, 4, 5]:

                k0 = i_icb
                k1 = i_cmb

            elif i_sample[j] in [6, 7]:
                
                k0 = 0 
                k1 = i_icb

            else:

                raise ValueError

            r_1d = eigfunc_dict['r'][k0 : k1]

            for key in keys:

                X_1d = eigfunc_dict[key][k0 : k1]
                eigfunc_interpolated[key][j] = np.interp(r_sample[j], r_1d, X_1d)

    elif mode_type in ['T', 'I']:

        if i_toroidal is None:

            if mode_type == 'T':

                i_sample_nonzero = [0, 1, 2]

            elif mode_type == 'I':

                i_sample_nonzero = [6, 7]

        elif i_toroidal == 0:

            i_sample_nonzero = [6, 7]

        elif i_toroidal == 1:

            i_sample_nonzero = [0, 1, 2]

        eigfunc_interpolated = dict()
        eigfunc_interpolated['W'] = np.zeros(n_radii)

        for j in range(n_radii):

            if i_sample[j] in i_sample_nonzero:

                eigfunc_interpolated['W'][j] = np.interp(r_sample[j],
                        eigfunc_dict['r'], eigfunc_dict['W'])

            else:

                eigfunc_interpolated['W'][j] = 0.0

    else:

        raise ValueError
    
    plot = False
    if plot:

        import matplotlib.pyplot as plt
        if mode_type in ['T', 'I']:

            keys = ['W']

        elif mode_type == 'R':

            keys = ['U']

        else:

            keys = ['U', 'V']

        for key in keys:
            
            plt.plot(eigfunc_dict['r'], eigfunc_dict[key], marker = '.')
            plt.plot(r_sample, eigfunc_interpolated[key], marker = '.')

        #plt.legend()
        plt.show()

    return eigfunc_interpolated

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref") 
    args = parser.parse_args()
    path_1d_input = args.path_ref

    # Get 1-D mode information.
    mode_info_1d, model_1d, run_info_1d = load_1d_mode_info(path_1d_input)

    # Read 3D input file.
    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii  = read_input_NMPostProcess()

    # Define directories.
    dir_processed = os.path.join(dir_NM, 'processed')

    # Get a list of 3-D modes.
    i_mode_list = get_list_of_modes_from_coeff_files(dir_NM, option) 
    num_modes = len(i_mode_list)

    # Get frequencies of 3-D modes.
    file_eigval_list        = os.path.join(dir_processed, 'eigenvalue_list.txt')
    _, f_3D = read_eigenvalues(file_eigval_list)

    # Do projection for each mode.
    first_iteration = True
    #for i_mode in i_mode_list:
    for i_mode in [1]:

        # Get the masses of the shells.
        if first_iteration:

            shell_mass = get_shell_mass_wrapper(dir_NM, model_1d)
            first_iteration = False
        
        # Do the projection.
        projection_wrapper_one_mode(dir_NM, i_mode, f_3D[i_mode - 1], mode_info_1d, run_info_1d, l_max, shell_mass)

    return

if __name__ == '__main__':

    main()
