import argparse
import os

import numpy as np

from NMPostProcess.common import (convert_complex_sh_to_real,
                            mkdir_if_not_exist,
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

    # Dictionary for saving mode type as integer.
    mode_type_to_int_dict = {'R' : 0, 'S' : 1, 'T' : 2, 'T1' : 2, 'I' : 3, 'T0' : 3}

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

    #mode_type_test = 'T1'
    #mode_info_1d = {mode_type_test : mode_info_1d[mode_type_test]}

    # Count the modes.
    num_modes_1d_by_type_no_multiplets = dict()
    num_modes_1d_by_type_with_multiplets = dict()
    num_modes_1d_total_no_multiplets = 0
    num_modes_1d_total_with_multiplets = 0
    # 
    for mode_type in mode_info_1d.keys():
        
        num_modes_1d_by_type_no_multiplets[mode_type] = len(mode_info_1d[mode_type]['l'])
        #
        num_modes_1d_total_no_multiplets = num_modes_1d_total_no_multiplets + \
                                        num_modes_1d_by_type_no_multiplets[mode_type]
        
        num_modes_1d_by_type_with_multiplets[mode_type] = 0
        for j in range(num_modes_1d_by_type_no_multiplets[mode_type]):
            
            l_j = mode_info_1d[mode_type]['l'][j]
            num_modes_1d_by_type_with_multiplets[mode_type] = \
                    num_modes_1d_by_type_with_multiplets[mode_type] + ((2 * l_j) + 1)

        num_modes_1d_total_with_multiplets = num_modes_1d_total_with_multiplets + \
                num_modes_1d_by_type_with_multiplets[mode_type]

    # Prepare to loop over all the 1-D modes and evaluate scalar product with the 3-D mode.
    #
    mode_type_out_array = np.zeros(num_modes_1d_total_with_multiplets, dtype = np.int)
    n                   = np.zeros(num_modes_1d_total_with_multiplets, dtype = np.int)
    l                   = np.zeros(num_modes_1d_total_with_multiplets, dtype = np.int)
    m                   = np.zeros(num_modes_1d_total_with_multiplets, dtype = np.int)
    product_real_part   = np.zeros(num_modes_1d_total_with_multiplets)
    product_imag_part   = np.zeros(num_modes_1d_total_with_multiplets)
    #
    i0 = 0

    # Loop over mode types.
    for mode_type_1d in mode_info_1d.keys():

        # Get mode type integer for saving output.
        mode_type_int = mode_type_to_int_dict[mode_type_1d]

        # Loop over modes of the given mode type.
        for j in range(num_modes_1d_by_type_no_multiplets[mode_type_1d]):

            print('Mode_type {:>2}, mode {:>4d} of {:>4d}'.format(mode_type_1d,
                    j + 1, num_modes_1d_by_type_no_multiplets[mode_type_1d]))

            # Get n, l and multiplicity of current 1-D mode.
            n_j = mode_info_1d[mode_type_1d]['n'][j]
            l_j = mode_info_1d[mode_type_1d]['l'][j]
            multiplicity_j = (2 * l_j) + 1

            # Get upper index in output array.
            i1 = i0 + multiplicity_j       

            # All modes in the multiplet have the same mode type.
            mode_type_out_array[i0 : i1] = mode_type_int

            # All (2l + 1) modes in the multiplet have the same n and l value.
            n[i0 : i1] = n_j
            l[i0 : i1] = l_j

            # Find the indices of the given l value in the coefficient list.
            k = np.where((l_coeffs == l_j))[0]

            # Get the m values from the coefficient list.
            m[i0 : i1] = m_coeffs[k]

            # Do projection of real part.
            product_real_part[i0 : i1] = project_one_mode_onto_one_multiplet(
                                    header['r_sample'], header['i_sample'],
                                    shell_mass,
                                    coeffs[0, :, k], coeffs[1, :, k],
                                    coeffs[2, :, k], run_info_1d, mode_type_1d,
                                    mode_info_1d[mode_type_1d]['n'][j],
                                    mode_info_1d[mode_type_1d]['l'][j],
                                    mode_info_1d[mode_type_1d]['f'][j])

            # Do projection of imaginary part.
            product_imag_part[i0 : i1] = project_one_mode_onto_one_multiplet(
                                    header['r_sample'], header['i_sample'],
                                    shell_mass,
                                    coeffs[3, :, k], coeffs[4, :, k],
                                    coeffs[5, :, k], run_info_1d, mode_type_1d,
                                    mode_info_1d[mode_type_1d]['n'][j],
                                    mode_info_1d[mode_type_1d]['l'][j],
                                    mode_info_1d[mode_type_1d]['f'][j])

            # Prepare for next iteration of loop.
            i0 = i1

    # Save output.
    out_fmt = '{:>2d} {:>5d} {:>5d} {:>+6d} {:>+19.12e} {:>+19.12e}\n'
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_projections = os.path.join(dir_processed, 'projections')
    mkdir_if_not_exist(dir_projections)
    file_out = 'mode_{:>05d}.txt'.format(i_mode)
    path_out = os.path.join(dir_projections, file_out)
    print('Writing to {:}'.format(path_out))
    with open(path_out, 'w') as out_id:

        for k in range(num_modes_1d_total_with_multiplets):

            out_id.write(out_fmt.format(
                    mode_type_out_array[k],
                    n[k], l[k], m[k],
                    product_real_part[k], product_imag_part[k]))

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

def project_one_mode_onto_one_multiplet(r_sample, i_sample, shell_mass, U_3d, V_3d, W_3d, run_info_1d, type_1d, n_1d, l_1d, f_1d):

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

    ##if (n_1d == 1) & (l_1d == 17):
    #if (n_1d == 2) & (l_1d == 17):

    #    import matplotlib.pyplot as plt
    #    fig, ax_arr = plt.subplots(1, 2, figsize = (11.0, 8.5), sharey = True)

    #    ax = ax_arr[0]

    #    ax.plot(eigfunc_interpolated['U'], r_sample, marker = '.', label = 'U (interpolated)')
    #    ax.plot(eigfunc_interpolated['V'], r_sample, marker = '.', label = 'V (interpolated)')

    #    i_compare = 0
    #    scale = np.max(np.abs(U_3d[i_compare, :])) / np.max(np.abs(eigfunc_interpolated['U']))
    #    ax.plot(U_3d[i_compare, :]/scale, r_sample, marker = '.', label = 'U')
    #    ax.plot(V_3d[i_compare, :]/scale, r_sample, marker = '.', label = 'V')

    #    ax = ax_arr[1]

    #    ax.plot(shell_mass *( U_3d[i_compare, :] * eigfunc_interpolated['U'] + V_3d[i_compare, :] * eigfunc_interpolated['V']),
    #                r_sample, marker = '.')

    #    ax.legend()

    #    for ax in ax_arr:

    #        ax.axvline()
    #    
    #    print(shell_mass)
    #    print(r_sample)
    #    plt.show()

    #    import sys
    #    sys.exit()

    # Evaluate the scalar product.
    num_in_multiplet, num_radii = U_3d.shape
    product = np.zeros(num_in_multiplet)
    for i in range(num_radii):
        
        if type_1d == 'R':

            product_i = (shell_mass[i] * U_3d[:, i] * eigfunc_interpolated['U'][i])

        elif type_1d == 'S': 

            product_i = shell_mass[i] * (U_3d[:, i] * eigfunc_interpolated['U'][i] +
                                         V_3d[:, i] * eigfunc_interpolated['V'][i])
            if (n_1d == 1) & (l_1d == 17):
                
                print('{:>5d} {:>+.3e}'.format(i, product_i[0]))

        elif type_1d[0] == 'T':

            product_i = (shell_mass[i] * W_3d[:, i] * eigfunc_interpolated['W'][i])

        product = product + product_i

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

def get_max_amp(mode_info_3d):

    # Find maximum excitation amplitude.
    max_amp = 0.0
    for mode_type in mode_info_3d.keys():
        
        if mode_info_3d[mode_type] is None:

            continue

        for multiplet in mode_info_3d[mode_type].keys():

            amp_list = np.sqrt( mode_info_3d[mode_type][multiplet]['coeff_real']**2.0 + \
                                mode_info_3d[mode_type][multiplet]['coeff_imag']**2.0)
            max_amp_i = np.max(amp_list)
            #print('{:} {:>15.3f}'.format(multiplet, 1.0E9*max_amp_i))
            if max_amp_i > max_amp:

                max_amp = max_amp_i

    return max_amp

def get_singlet_amplitude(mode_info_3d, max_amp):

    singlet_amplitude = dict()

    for mode_type in mode_info_3d.keys():

        if mode_info_3d[mode_type] is None:

            continue

        for nl_key in mode_info_3d[mode_type].keys():

            # Get string representing mode type, n and l.
            type_nl_key = '{:}_{:}'.format(mode_type, nl_key)

            # Unpack dictionary.
            x_real = mode_info_3d[mode_type][nl_key]['coeff_real']
            x_imag = mode_info_3d[mode_type][nl_key]['coeff_imag']

            singlet_amplitude[type_nl_key] = \
                np.sqrt(np.sum(x_real**2.0 + x_imag**2.0)) / max_amp

    return singlet_amplitude

def load_3d_mode_info(path_in):

    # Dictionary for loading mode type saved as integer.
    mode_type_from_int_dict = {0 : 'R', 1 : 'S', 2 : 'T1', 3 : 'T0'}

    # Load data and extract columns.
    data = np.loadtxt(path_in)
    mode_type_int   = data[:, 0].astype(np.int)
    n               = data[:, 1].astype(np.int)
    l               = data[:, 2].astype(np.int)
    m               = data[:, 3].astype(np.int)
    coeff_real      = data[:, 4]
    coeff_imag      = data[:, 5]

    # Separate different mode types.
    mode_info = dict()
    mode_type_int_list = np.sort(np.unique(mode_type_int))
    for mode_type_int_i in mode_type_int_list:

        # Get mode type string.
        mode_type = mode_type_from_int_dict[mode_type_int_i]
        mode_info[mode_type] = dict()

        # Find modes with the given mode type.
        j = np.where(mode_type_int_i == mode_type_int)[0]
        
        # Store select modes in dictionary.
        mode_info[mode_type]['n'] = n[j]
        mode_info[mode_type]['l'] = l[j]
        mode_info[mode_type]['m'] = m[j]
        mode_info[mode_type]['coeff_real'] = coeff_real[j]
        mode_info[mode_type]['coeff_imag'] = coeff_imag[j]

    # Group modes into specific multiplets.
    mode_info_grouped = dict()
    for mode_type in mode_info.keys():

        num_modes = len(mode_info[mode_type]['n'])

        mode_info_grouped[mode_type] = dict()

        for i in range(num_modes):

            # Unpack dictionary.
            n = mode_info[mode_type]['n'][i]
            l = mode_info[mode_type]['l'][i]
            #m = mode_info[mode_type]['m'][i]
            #coeff_real = mode_info[mode_type][coeff_real][i]
            #coeff_imag = mode_info[mode_type][coeff_imag][i]

            # Get unique string identifying multiplet.
            nl_key = '{:>05d}_{:>05d}'.format(n, l)

            if nl_key in mode_info_grouped[mode_type].keys():
                
                for variable in ['m', 'coeff_real', 'coeff_imag']:

                    mode_info_grouped[mode_type][nl_key][variable] = \
                        np.append(mode_info_grouped[mode_type][nl_key][variable],
                                    mode_info[mode_type][variable][i])

            else:

                mode_info_grouped[mode_type][nl_key] = dict()
                mode_info_grouped[mode_type][nl_key]['n'] = n
                mode_info_grouped[mode_type][nl_key]['l'] = l 

                for variable in ['m', 'coeff_real', 'coeff_imag']:

                    mode_info_grouped[mode_type][nl_key][variable] = \
                            np.atleast_1d(mode_info[mode_type][variable][i])

    # Sort the multiplets by m-value.
    for mode_type in mode_info_grouped.keys():

        for nl_key in mode_info_grouped[mode_type]:

            i_sort = np.argsort(mode_info_grouped[mode_type][nl_key]['m'])
            for variable in ['m', 'coeff_real', 'coeff_imag']:

                mode_info_grouped[mode_type][nl_key][variable] = \
                        mode_info_grouped[mode_type][nl_key][variable][i_sort]

    # Fill in empty mode types.
    for mode_type in ['R', 'S', 'T0', 'T1']:

        if mode_type not in mode_info_grouped.keys():

            mode_info_grouped[mode_type] = None

    #return mode_info
    return mode_info_grouped

def report_wrapper(dir_processed, mode_info_1d, i_mode_list):
    
    path_report = os.path.join(dir_processed, 'projections', 'report.txt')
    print("Writing to {:}".format(path_report))
    with open(path_report, 'w') as out_id:

        for i_mode in i_mode_list:

            report_str = report(dir_processed, mode_info_1d, i_mode)

            out_id.write(report_str + '\n')

    return

def report(dir_processed, mode_info_1d, i_mode, amp_thresh = 0.2):

    # Find projections directory.
    dir_projections = os.path.join(dir_processed, 'projections')

    # Load major modes.
    n_singlet, l_singlet, f_singlet, amp_singlet, mode_type_list = \
        get_major_modes(dir_projections, mode_info_1d, i_mode, amp_thresh)
    num_singlet = len(n_singlet)
    
    report_str = ''
    for i in range(num_singlet):
        
        if i > 0:
            
            report_str_i = '{:>2} {:>3d} {:>3d} {:>7.3f}    '.format(mode_type_list[i], n_singlet[i],
                    l_singlet[i], amp_singlet[i])

        else:

            report_str_i = '{:>5d} {:>2} {:>3d} {:>3d} {:>7.3f}    '.format(i_mode, mode_type_list[i], n_singlet[i],
                    l_singlet[i], amp_singlet[i])
        report_str = report_str + report_str_i

    print(report_str)
    
    return report_str

def get_major_modes(dir_projections, mode_info_1d, i_mode, amp_thresh):

    # Get projection coefficients of 3-D modes.
    file_projection = 'mode_{:>05d}.txt'.format(i_mode)
    path_projection = os.path.join(dir_projections, file_projection)
    mode_info_3d = load_3d_mode_info(path_projection)

    # Get maximum amplitude.
    max_amp = get_max_amp(mode_info_3d)

    # Get the amplitude in each singlet.
    singlet_amplitude = get_singlet_amplitude(mode_info_3d, max_amp)

    # Count the number of singlets.
    num_singlets = 0
    for type_nl_key in singlet_amplitude.keys():
        
        num_singlets = num_singlets + 1

    # Prepare plotting arrays.
    mode_type_list = []
    n_singlet   = np.zeros(num_singlets, dtype = np.int)
    l_singlet   = np.zeros(num_singlets, dtype = np.int)
    f_singlet   = np.zeros(num_singlets)
    amp_singlet = np.zeros(num_singlets)

    # Fill plotting arrays.
    for i, type_nl_key in enumerate(singlet_amplitude.keys()):

        mode_type, n_str, l_str = type_nl_key.split('_')
        n_singlet[i] = int(n_str)
        l_singlet[i] = int(l_str)

        k = np.where((mode_info_1d[mode_type]['n'] == n_singlet[i]) &
                     (mode_info_1d[mode_type]['l'] == l_singlet[i]))[0]
        
        f_singlet[i]   = mode_info_1d[mode_type]['f'][k]
        amp_singlet[i] = singlet_amplitude[type_nl_key]

        mode_type_list.append(mode_type)

    # Normalise amplitude.
    total_amp = np.sqrt(np.sum(amp_singlet**2.0))
    amp_singlet = amp_singlet/total_amp

    # Filter small amplitudes.
    i_thresh = np.where(amp_singlet > amp_thresh)[0]
    n_singlet   = n_singlet[i_thresh]
    l_singlet   = l_singlet[i_thresh]
    f_singlet   = f_singlet[i_thresh]
    amp_singlet = amp_singlet[i_thresh]
    mode_type_list = [mode_type_list[i] for i in i_thresh]

    # Count filtered list.
    num_singlet = len(l_singlet)

    # Sort by amplitude.
    i_sort = np.argsort(-amp_singlet)
    n_singlet   = n_singlet[i_sort]
    l_singlet   = l_singlet[i_sort]
    f_singlet   = f_singlet[i_sort]
    amp_singlet = amp_singlet[i_sort]
    mode_type_list = [mode_type_list[i] for i in i_sort]

    #i_sort = np.argsort(f_singlet)
    #n_singlet   = n_singlet[i_sort]
    #l_singlet   = l_singlet[i_sort]
    #f_singlet   = f_singlet[i_sort]
    #amp_singlet = amp_singlet[i_sort]
    #mode_type_list = [mode_type_list[i] for i in i_sort]
    ##
    #i_sort = np.argsort(l_singlet)
    #n_singlet   = n_singlet[i_sort]
    #l_singlet   = l_singlet[i_sort]
    #f_singlet   = f_singlet[i_sort]
    #amp_singlet = amp_singlet[i_sort]
    #mode_type_list = [mode_type_list[i] for i in i_sort]

    return n_singlet, l_singlet, f_singlet, amp_singlet, mode_type_list

def main():

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref") 
    parser.add_argument("--print_report", action = 'store_true', help = 'Run the command a second time using this flag to report the results of the projection.')
    args = parser.parse_args()
    path_1d_input = args.path_ref
    print_report = args.print_report

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
    if print_report:
        
        report_wrapper(dir_processed, mode_info_1d, i_mode_list)

    else:

        first_iteration = True
        for i_mode in i_mode_list:

            print('Mode {:>5d}'.format(i_mode))

            # Get the masses of the shells.
            if first_iteration:

                shell_mass = get_shell_mass_wrapper(dir_NM, model_1d)
                shell_mass = shell_mass[::-1]
                first_iteration = False
            
            # Do the projection.
            projection_wrapper_one_mode(dir_NM, i_mode, f_3D[i_mode - 1], mode_info_1d, run_info_1d, l_max, shell_mass)

    return

if __name__ == '__main__':

    main()
