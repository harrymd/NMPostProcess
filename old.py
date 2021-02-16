def get_indices_of_regions3(nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default', ellipticity_profile = None):
    '''
    For each region (outer surface, interior, and inner surface of each shell), find the indices of the samples belonging to that region.

    Input:

    See 'Definitions of variables.'

    Output:

    See 'Definitions of variables.'
    '''
    
    # Get the radius of the surface and the number of discontinuities.
    r_surface = r_discons[0]
    n_discons = len(r_discons)

    # Get the coordinates of the samples.
    sample_nodes = nodes[node_idxs, :]

    # Get the radial coordinates of the samples.
    r_nodes = np.linalg.norm(nodes, axis = 1)
    r_samples = r_nodes[node_idxs]

    # Find nodes on the free surface.
    i_shell = 0
    is_outer = True 
    surface_condition, j_surface, surface_conv_hull = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, ellipticity_profile = ellipticity_profile, boundary_tol = 1.5, surface_method = 'radius')

    # Find all "interior" nodes which do not belong to any interface (must remove the surface nodes).
    # Note: Assume higher-order nodes (3, 4) have already been removed.
    interior_condition = (((node_attbs == 0) | (node_attbs == 1)) & ~surface_condition)
    #interior_condition = (((node_attbs == 0) | (node_attbs == 1) | (node_attbs == 3) | (node_attbs == 4)) & ~surface_condition)
    j_interior_sample = np.where(interior_condition)[0]
    
    # Case 1: There are no interior discontinuities.
    if n_discons == 1:
        
        # Find samples inside the object.
        index_lists_interior = [j_interior_sample]
        
        # Find samples on the surface of the object.
        index_lists_boundaries = [j_surface]

    # Case 2: There are some interior discontinuities.
    else:

        # Make a list of convex hulls, one for each discontinuity.
        #conv_hull_list = [surface_conv_hull]
        #conv_hull_list = []
        in_hull_list = []
        is_outer = True
        for i_shell in range(1, n_discons):

            _, j_discon, _ = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')
            i_j_discon = node_idxs[j_discon]
            nodes_discon = nodes[i_j_discon, :]
            buff = 1.0E-5
            delaunay = Delaunay(nodes_discon*(1.0 + buff))
            #conv_hull = ConvexHull(nodes_discon)
            #conv_hull = delaunay
            #conv_hull_list.append(conv_hull)

            in_hull_cond = in_hull(sample_nodes, delaunay)
            in_hull_list.append(in_hull_cond)

        # Initialise lists.
        index_lists_interior = []
        index_lists_boundaries = [j_surface]

        ## Make a list of boundaries of shells (r_surface, r1, r2, ..., 0.0).
        #r_discons_bounded = np.concatenate([r_discons, [0.0]])

        # Find the indices of interior samples in each shell.
        for i_shell in range(n_discons):
            
            ## Find samples at the appropriate radial distance.
            #radius_condition = ((r_samples < r_discons_bounded[i_shell]) & (r_samples > r_discons_bounded[i_shell + 1]))

            # Find condition of outer convex hull.
            if i_shell == 0:

                # All points lie within the outermost convex hull.
                outer_hull_condition = True

            else:

                outer_hull_condition = in_hull_list[i_shell - 1]

            # Find condition of inner convex_hull.
            if i_shell == (n_discons - 1):

                # All points lie outside the innermost hull (point 0, 0, 0).
                inner_hull_condition = False

            else:

                inner_hull_condition = in_hull_list[i_shell]

            # Apply the radial condition and the interior node attribute condition.
            shell_condition = (~inner_hull_condition & outer_hull_condition & interior_condition)
            j_shell = np.where(shell_condition)[0]
            
            # Store.
            index_lists_interior.append(j_shell)

        # Find the indices of boundary samples in each shell.
        for i_shell in range(n_discons):

            # Loop over the outer and inner sides of each shell.
            # The outer surface of the outer shell has already been found.
            if i_shell == 0:

                is_outer_list = [False]

            elif (i_shell < (n_discons - 1)):

                is_outer_list = [True, False]

            # The innermost shell does not have an inner side.
            else:

                is_outer_list = [True]

            for is_outer in is_outer_list:
                
                # Search for the boundary samples.
                _, j_discon, _ = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')

                index_lists_boundaries.append(j_discon)

    # Merge the lists into a single list.
    # The list starts with the free surface and then the first interior region.
    index_lists = [index_lists_boundaries[0], index_lists_interior[0]]
    if len(index_lists_interior) > 1:

        for i in range(1, len(index_lists_interior)):

            # Between every interior region, there are two boundaries.
            index_lists.append(index_lists_boundaries[2*i - 1])
            index_lists.append(index_lists_boundaries[2*i])

            # The next interior region.
            index_lists.append(index_lists_interior[i])

    for index_list in index_lists:

        print(len(index_list))

    return index_lists_interior, index_lists_boundaries, index_lists

def get_indices_of_regions_old2(nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default'):
    '''
    For each region (outer surface, interior, and inner surface of each shell), find the indices of the samples belonging to that region.

    Input:

    See 'Definitions of variables.'

    Output:

    See 'Definitions of variables.'
    '''
    
    # Get the radius of the surface and the number of discontinuities.
    r_surface = r_discons[0]
    n_discons = len(r_discons)

    print(r_surface)
    print(n_discons)
    print('nodes.shape', nodes.shape)
    print('node_attbs.shape', node_attbs.shape)
    print('node_idxs.shape', node_idxs.shape)
    print(state_outer)

    # Get the radial coordinates of the samples.
    r_nodes = np.linalg.norm(nodes, axis = 1)
    r_samples = r_nodes[node_idxs]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = plt.gca()

    j_0 = np.where(node_attbs == 0)[0]
    j_1 = np.where(node_attbs == 1)[0]
    j_2 = np.where(node_attbs == 2)[0]
    j_6 = np.where(node_attbs == 6)[0]

    for i in np.unique(node_attbs):
        
        j = np.where(node_attbs == i)[0]
        print(i, len(j))

    #ax.hist(r_samples[j_0])
    #ax.hist(r_samples[j_1])
    #ax.hist(r_samples[j_2])
    #ax.hist(r_samples[j_6])
    ax.hist(r_samples[r_samples <= 1221.5])

    print(np.sum(r_samples < 1050.0))


    plt.show()

    # Find nodes on the free surface.
    i_shell = 0
    is_outer = True
    surface_condition, j_surface = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer)

    print('j_surface.shape', j_surface.shape)

    # Find all "interior" nodes which do not belong to any interface (must remove the surface nodes).
    # Note: Assume higher-order nodes (3, 4) have already been removed.
    interior_condition = (((node_attbs == 0) | (node_attbs == 1)) & ~surface_condition)
    #interior_condition = (((node_attbs == 0) | (node_attbs == 1) | (node_attbs == 3) | (node_attbs == 4)) & ~surface_condition)
    j_interior_sample = np.where(interior_condition)[0]
    
    print(np.sum(surface_condition))
    print('j_interior_sample.shape', j_interior_sample.shape)
    print(np.unique(node_attbs))

    # Case 1: There are no interior discontinuities.
    if n_discons == 1:
        
        # Find samples inside the object.
        index_lists_interior = [j_interior_sample]
        
        # Find samples on the surface of the object.
        index_lists_boundaries = [j_surface]

    # Case 2: There are some interior discontinuities.
    else:

        # Initialise lists.
        index_lists_interior = []
        index_lists_boundaries = [j_surface]

        # Make a list of boundaries of shells (r_surface, r1, r2, ..., 0.0).
        r_discons_bounded = np.concatenate([r_discons, [0.0]])

        print(r_discons_bounded)


        # Find the indices of interior samples in each shell.
        for i_shell in range(n_discons):
            
            # Find samples at the appropriate radial distance.
            radius_condition = ((r_samples < r_discons_bounded[i_shell]) & (r_samples > r_discons_bounded[i_shell + 1]))

            print('\n')
            print(i_shell)
            print(r_discons_bounded[i_shell])
            print(r_discons_bounded[i_shell + 1])

            # Apply the radial condition and the interior node attribute condition.
            shell_condition = (radius_condition & interior_condition)
            j_shell = np.where(shell_condition)[0]
            
            # Store.
            index_lists_interior.append(j_shell)

        # Find the indices of boundary samples in each shell.
        for i_shell in range(n_discons):

            # Loop over the outer and inner sides of each shell.
            # The outer surface of the outer shell has already been found.
            if i_shell == 0:

                is_outer_list = [False]

            elif (i_shell < (n_discons - 1)):

                is_outer_list = [True, False]

            # The innermost shell does not have an inner side.
            else:

                is_outer_list = [True]

            for is_outer in is_outer_list:
                
                # Search for the boundary samples.
                _, j_discon = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')

                index_lists_boundaries.append(j_discon)

    # Merge the lists into a single list.
    # The list starts with the free surface and then the first interior region.
    index_lists = [index_lists_boundaries[0], index_lists_interior[0]]
    if len(index_lists_interior) > 1:

        for i in range(1, len(index_lists_interior)):

            # Between every interior region, there are two boundaries.
            index_lists.append(index_lists_boundaries[2*i - 1])
            index_lists.append(index_lists_boundaries[2*i])

            # The next interior region.
            index_lists.append(index_lists_interior[i])

    print(len(index_lists))
    a = 0
    for index_list in index_lists:

        print(len(index_list))
        a = a + len(index_list)

    print(a)

    return index_lists_interior, index_lists_boundaries, index_lists

def get_indices_of_regions_old1(nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default'):
    '''
    For each region (outer surface, interior, and inner surface of each shell), find the indices of the samples belonging to that region.

    Input:

    See 'Definitions of variables.'

    Output:

    See 'Definitions of variables.'
    '''
    
    # Get the radius of the surface and the number of discontinuities.
    r_surface = r_discons[0]
    n_discons = len(r_discons)

    # Get the radial coordinates of the samples.
    r_nodes = np.linalg.norm(nodes, axis = 1)
    r_samples = r_nodes[node_idxs]

    # Find nodes on the free surface.
    i_shell = 0
    is_outer = True
    surface_condition, j_surface = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer)

    # Find all interior nodes (must remove the surface nodes).
    interior_condition = (((node_attbs == 0) | (node_attbs == 1)) & ~surface_condition)
    j_interior_sample = np.where(interior_condition)[0]

    # Case 1: There are no interior discontinuities.
    if n_discons == 1:
        
        # Find samples inside the object.
        index_lists_interior = [j_interior_sample]
        
        # Find samples on the surface of the object.
        index_lists_boundaries = [j_surface]

    # Case 2: There are some interior discontinuities.
    else:

        # Initialise lists.
        index_lists_interior = []
        index_lists_boundaries = [j_surface]

        # Make a list of boundaries of shells (r_surface, r1, r2, ..., 0.0).
        r_discons_bounded = np.concatenate([r_discons, [0.0]])

        # Find the indices of interior samples in each shell.
        for i_shell in range(n_discons):
            
            # Find samples at the appropriate radial distance.
            radius_condition = ((r_samples < r_discons_bounded[i_shell]) & (r_samples > r_discons_bounded[i_shell + 1]))

            # Apply the radial condition and the interior node attribute condition.
            shell_condition = (radius_condition & interior_condition)
            j_shell = np.where(shell_condition)[0]
            
            # Store.
            index_lists_interior.append(j_shell)

        # Find the indices of boundary samples in each shell.
        for i_shell in range(n_discons):

            # Loop over the outer and inner sides of each shell.
            # The outer surface of the outer shell has already been found.
            if i_shell == 0:

                is_outer_list = [False]

            elif (i_shell < (n_discons - 1)):

                is_outer_list = [True, False]

            # The innermost shell does not have an inner side.
            else:

                is_outer_list = [True]

            for is_outer in is_outer_list:
                
                # Search for the boundary samples.
                _, j_discon = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default')

                index_lists_boundaries.append(j_discon)

    # Merge the lists into a single list.
    # The list starts with the free surface and then the first interior region.
    index_lists = [index_lists_boundaries[0], index_lists_interior[0]]
    if len(index_lists_interior) > 1:

        for i in range(1, len(index_lists_interior)):

            # Between every interior region, there are two boundaries.
            index_lists.append(index_lists_boundaries[2*i - 1])
            index_lists.append(index_lists_boundaries[2*i])

            # The next interior region.
            index_lists.append(index_lists_interior[i])

    return index_lists_interior, index_lists_boundaries, index_lists

def old_get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, boundary_tol = 'default', surface_method = 'convhull', ellipticity_profile = None):
    '''
    Finds the samples of the eigvector which belong to a specified boundary.
    
    Input:

    See 'Definitions of variables.'

    surface_method
        A string specifying the method used to identify the samples on the outer surface. Currently, the supported options are
            'convhull'  (Default.) Finds the nodes belonging to the convex hull of the mesh. This only yields all of the surface nodes if the surface mesh is convex.
            'radius'    Uses the outer surface radius to find the samples on the surface. This only works if the surface is spherical.

    Output:

    condition
        (n_samples) True for the samples on the boundary, False otherwise. Therefore j_boundary = np.where(condition)[0].
    j_boundary
        The indices of the samples which belong to the boundary.
    '''

    conv_hull = None

    # Case 1: The interface is the outer surface.
    if (i_shell == 0) and is_outer:

        # Find the points on the surface using the convex hull.
        # This method only works if the surface is convex.
        if surface_method == 'convhull':
        
            # Build complex hull.
            conv_hull = ConvexHull(nodes)
            hull_indices = conv_hull.simplices

            hull_indices = np.unique(hull_indices.flatten())

            condition = np.array([i in hull_indices for i in node_idxs], dtype = np.int)

        # Find the points on the surface based on their radial coordinate.
        # This method only works if the surface is a sphere.
        elif surface_method == 'radius':

            # Find radius of outer surface.
            r_surface = r_discons[0]

            # Set default value of comparison tolerance for boundaries.
            if boundary_tol == 'default':

                boundary_tol = r_surface*1.0E-7
            
            if ellipticity_profile is None:

                r_samples = np.linalg.norm(nodes[node_idxs], axis = 1)

            else:

                r_nodes, _ = XYZ_to_REll(*nodes.T, *ellipticity_profile.T)
                r_samples = r_nodes[node_idxs]
                
                #import matplotlib.pyplot as plt

                #fig = plt.figure()
                #ax = plt.gca()

                #bins = np.linspace(6300, 6375, num = 76)

                #plt.hist(r_nodes, bins = bins)

                #ax.set_yscale('log')

                #plt.show()

            r_min = r_surface - boundary_tol
            condition = (r_samples > r_min)

        else:

            raise ValueError('Method for detecting surface nodes ({:}) was not recognised. Options are \'convhull\' and \'radius\'')

    # Case 2: The interface is an internal fluid-solid boundary.
    else:
        
        # Find the index of the region the sample belongs to.
        if is_outer:
            
            i_region_discon = i_shell

        else:

            i_region_discon = i_shell + 1

        # Get discon lists.
        r_discon_midpoints, state_list, n_interior_discons = get_discon_info(r_discons, state_outer)
        
        if n_interior_discons > 1:

            # Find which nodes are close to the specified discontinuity. 
            r_nodes = np.linalg.norm(nodes, axis = 1)
            r_samples = r_nodes[node_idxs]
            radius_condition = ((r_samples < r_discon_midpoints[i_region_discon - 1]) & (r_samples > r_discon_midpoints[i_region_discon]))

        else:
            
            # If there is only one internal discontinuity, we do not have to specify a radius condition to identify it.
            radius_condition = True

        # Case 2a. The query point is in a solid shell.
        if state_list[i_shell] == 'solid':

            attb_condition = (node_attbs == 2)

        # Case 2b. The query point is in a liquid shell.
        else:

            attb_condition = (node_attbs == 6)

        # Apply the attribute and radius conditions simultaneously.
        condition = (attb_condition & radius_condition)
        
    j_boundary = np.where(condition)[0]

    return condition, j_boundary, conv_hull

def main2():

    dir_PM, dir_NM, option, l_max, i_mode_str, n_radii = read_input_NMPostProcess()

    # Read the global configuration file.
    path_global_conf = os.path.join(dir_NM, 'global_conf')
    job = read_global_conf(path_global_conf)

    # Determine whether the modes are real or complex.
    job_to_real_or_complex_dict = {2 : 'real', 6 : 'complex'}
    real_or_complex = job_to_real_or_complex_dict[job]

    # Check for ellipticity profile.
    path_ellipticity = os.path.join(dir_PM, 'input', 'ellipticity_profile.txt')
    try:

        ellipticity_profile = np.loadtxt(path_ellipticity)
        ellipticity_profile[:, 0] = ellipticity_profile[:, 0]*1.0E-3 # km to m.

    except OSError:

        print('No ellipticity profile found at {:}\nAssuming spherical geometry.'.format(path_ellipticity))
        ellipticity_profile = None

    # Create the output directories if they do not exist.
    dir_processed = os.path.join(dir_NM, 'processed')
    dir_spectral = os.path.join(dir_processed, 'spectral')
    for dir_ in [dir_processed, dir_spectral]:
        mkdir_if_not_exist(dir_)
    
    # Write the nodes file if it doesn't already exist.
    # This file is simply a re-written version of some of the TetGen output, for convenience.
    path_nodes = os.path.join(dir_processed, 'nodes.txt')
    #if not os.path.exists(path_nodes):
    if True:
        
        #nodes, tets, tet_means, tet_attrib, neighs = read_mesh(file_base)
        nodes, node_bd_flag, _, _, _, neighs, face_list, face_attb, face_tets = read_mesh(dir_PM)
        print('Saving node file {:}'.format(path_nodes))
        np.savetxt(path_nodes, nodes)
    
    else:
        
        print('Node file already exists: {:}. Skipping creation.'.format(path_nodes))
        nodes = np.loadtxt(path_nodes)
    
    # Write the region index files if they don't already exist.
    path_index_lists = os.path.join(dir_processed, 'index_lists.txt')
    #if not os.path.exists(path_index_lists):
    if  True:
    
        # Read the discontinuity radius information.
        path_discon_info = os.path.join(dir_PM, 'radii.txt')
        r_discons, state_outer = read_discon_file(path_discon_info)
    
        # Load sample index and attribute information.
        print(dir_NM)
        node_idxs, node_attbs, i_first_order = load_sample_indices_and_attribs(dir_NM)

        # Get the radius of the surface and the number of discontinuities.
        r_surface = r_discons[0]
        n_discons = len(r_discons)

        # Get the coordinates of the samples.
        sample_nodes = nodes[node_idxs, :]

        # Get the radial coordinates of the samples.
        r_nodes = np.linalg.norm(nodes, axis = 1)
        r_samples = r_nodes[node_idxs]

        # Find nodes on the free surface.
        #i_shell = 0
        #is_outer = True 
        #surface_condition, j_surface, surface_conv_hull = get_samples_on_boundary(i_shell, is_outer, nodes, node_attbs, node_idxs, r_discons, state_outer, ellipticity_profile = ellipticity_profile, boundary_tol = 1.5, surface_method = 'radius')
    
    #print(node_idxs.shape)
    #print(r_nodes.shape)
    #print(nodes.shape)
    #print(node_bd_flag.shape)
    #print(np.unique(node_bd_flag))

    #print(r_discons)

    #print(face_list.shape)
    #print(face_attb.shape)
    #print(face_tets.shape)

    #print(face_list[-10:, :])
    #print(face_attb[-10:])
    #print(face_tets[-10:, :])
    
    n_faces = face_list.shape[0]
    i_bd_node = set()
    for i in range(n_faces):
        
        if (face_tets[i, 0] == -1) | (face_tets[i, 1] == -1):
            
            for j in range(3):

                i_bd_node.add(face_list[i, j])

    i_shell = 0
    for i_shell in [0, 1, 2]:

        #surface_condition, j_surface = 
        get_samples_on_boundary(i_shell, r_discons, r_nodes, i_bd_node, node_idxs)

    #print(np.max(neighs.flatten()))

    #i_bd = np.where(node_bd_flag == 1)[0]

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.gca()

    #ax.hist(r_nodes[i_bd])

    #plt.show()

    return
