"""utilities for efficiently working with patches of cortex (aka subsurfaces)"""
import numpy as np
import scipy.sparse

from .misc import _memo


class SubsurfaceMixin(object):
    """mixin for Surface of efficient methods for working with subsurfaces


    - see pycortex documentation for example usage


    Use Cases
    ---------
    - performing many operations on a subset of cortex
    - finding patches/paths in cortical surface (see Performance Characteristics)


    Performance Characteristics
    ---------------------------
    - main use case is faster implementation of geodesic_distance()
    - original geodesic_distance:
        - large startup cost (~10 s)
        - small subsequent cost (~200 ms)
        - use case: performing many repeated operations on large subsets of cortex
    - subsurface geodesic_distance:
        - cost is based on radius
            - 5 mm -> (~40 ms startup cost)
            - 25 mm -> (~200 ms startup cost)
        - use cases: calling operations small number of times or on medium subsets of cortex
    - [benchmarks recorded on lab desktop workstation]
    """

    def create_subsurface(self, vertex_mask=None, polygon_mask=None):
        """Create subsurface for efficient operations on subset of Surface

        - should specify either vertex_mask or polygon_mask
        - input vertex_mask is not necessarily the final vertex_mask used
            - final vertex_mask is always derived from polygon_mask
            - this prevents dangling vertices

        Parameters
        ----------
        - vertex_mask : boolean array
            - mask of which vertices to include
        - polygon_mask : boolean array
            - mask of which polygons to include
        """

        if polygon_mask is None:

            if vertex_mask is None:
                raise Exception('must specify vertex_mask or polygon_mask')

            polygon_mask = (
                vertex_mask[self.polys[:, 0]]
                * vertex_mask[self.polys[:, 1]]
                * vertex_mask[self.polys[:, 2]]
            )

        # select only vertices that appear in a polygon of polygon_mask
        vertex_mask = np.zeros(self.pts.shape[0], dtype=bool)
        vertex_mask[self.polys[polygon_mask].flat] = True

        # build map from old index to new index
        # vertices not in the subsurface are represented with large numbers
        vertex_map = np.ones(self.pts.shape[0], dtype=int) * np.iinfo(np.int32).max
        vertex_map[vertex_mask] = range(vertex_mask.sum())

        # reindex vertices and polygons
        subsurface_vertices = self.pts[vertex_mask, :]
        subsurface_polygons = vertex_map[self.polys[polygon_mask, :]]

        # create subsurface
        subsurface = self.__class__(pts=subsurface_vertices, polys=subsurface_polygons)
        subsurface.subsurface_vertex_mask = vertex_mask
        subsurface.subsurface_vertex_map = vertex_map
        subsurface.subsurface_polygon_mask = polygon_mask

        return subsurface

    @property
    @_memo
    def subsurface_vertex_inverse(self):
        return np.nonzero(self.subsurface_vertex_mask)[0]

    def get_connected_vertices(self, vertex, mask, old_version=False):
        """return vertices connected to vertex that satisfy mask

        - helper method for other methods

        Parameters
        ----------
        - vertex : one of [scalar int index | list of int indices | numpy array of int indices]
            vertex or set of vertices to use as seed
        - mask : boolean array
            mask of allowed neighbors
        - old_version : boolean (default=False)
            True = Use vertex adjacency to select patch (can cause errors in odd situations)
            False = Use poly adjacency to select patch (solves problem where a single edge but
            no polys connect two regions within the patch, makes geodesic distance errors)
        """
        n_vertices = self.pts.shape[0]
        n_polys = self.polys.shape[0]
        output_mask = np.zeros(n_vertices, dtype=bool)

        if np.issubdtype(type(vertex), np.integer):
            add_next = [vertex]
            output_mask[vertex] = True
        elif (
            isinstance(vertex, list)
            or (isinstance(vertex, np.ndarray) and np.issubdtype(vertex.dtype, np.integer))
        ):
            add_next = vertex
            output_mask[vertex] = True
        else:
            raise Exception('unknown vertex type:' + str(vertex))

        if old_version:
            while len(add_next) > 0:
                check = np.zeros(n_vertices, dtype=bool)
                check[self.adj[add_next, :].indices] = True
                add_next = check * mask * (~output_mask)
                output_mask[add_next] = True
                add_next = np.nonzero(add_next)[0]
        else:
            while len(add_next) > 0:
                check = np.zeros(n_vertices, dtype=bool)
                # Instead of just adjacent vertices, get adjacent polys
                check_polys = self.connected[add_next,:].indices
                # Will be checking within mask in this step for all verts for a poly being in the mask
                good_polys = check_polys[np.all(mask[self.polys[check_polys,:]], axis=1)]
                # Then get all verts from the good polys
                good_verts = np.unique(self.polys[good_polys])
                check[good_verts] = True
                # Mask is already used in selecting checked ones
                add_next = check * (~output_mask)
                output_mask[add_next] = True
                add_next = np.nonzero(add_next)[0]

        return output_mask

    def get_euclidean_patch(self, vertex, radius, old_version=False):
        """return connected vertices within some 3d euclidean distance of a vertex

        Parameters
        ----------
        - vertex : one of [scalar int index | list of int indices | numpy array of int indices]
            vertex or set of vertices to use as seed
        - radius : number
            distance threshold
        - old_version : boolean (default=False)
            True = Use vertex adjacency to select patch (can cause errors in odd situations)
            False = Use poly adjacency to select patch (solves problem where a single edge but
            no polys connect two regions within the patch, makes geodesic distance errors)
        """

        if np.issubdtype(type(vertex), np.integer):
            close_enough = self.get_euclidean_ball(self.pts[vertex, :], radius)
        elif (
            isinstance(vertex, list)
            or (isinstance(vertex, np.ndarray) and np.issubdtype(vertex.dtype, np.integer))
        ):
            mask_list = [self.get_euclidean_ball(self.pts[index, :], radius) for index in vertex]
            close_enough = np.array(mask_list).sum(axis=0).astype(bool)
        else:
            raise Exception('unknown vertex type: ' + str(type(vertex)))

        return {
            'vertex_mask': self.get_connected_vertices(vertex=vertex, mask=close_enough, old_version=old_version),
        }

    def get_euclidean_ball(self, xyz, radius):
        """return vertices within some 3d euclidean distance of an xyz coordinate

        Parameters
        ----------
        - xyz : array of shape (3,)
            center of euclidean ball
        - radius : number
            radius of euclidean ball
        """

        # unoptimized version:
        # distances = ((surface.pts - xyz) ** 2).sum(1) ** 0.5
        # return distances < radius

        # optimized version:
        diff = self.pts - xyz
        diff **= 2
        diff = diff.dot(np.ones(diff.shape[1]))  # precision fine because only summing 3 values
        diff **= 0.5

        return diff < radius

    def get_geodesic_patch(self, vertex, radius, attempts=5, m=1.0, old_version=False):
        """return vertices within some 2d geodesic distance of a vertex (or vertices)

        Parameters
        ----------
        - vertex : int
            index (or list of int indices) of seed vertex (or vertices)
        - radius : number
            radius to use as threshold
        - attempts : int
            number of attempts to use for working with singular subsurfaces
        - m : number
            reverse Euler step length, passed to geodesic_distance
        - old_version : boolean (default=False)
            True = Use vertex adjacency to select patch (can cause errors in odd situations)
            False = Use poly adjacency to select patch (solves problem where a single edge but
            no polys connect two regions within the patch, makes geodesic distance errors)

        Output
        ------
        - 'vertex_mask' : boolean mask of selected vertices
        - 'geodesic_distance' : array of geodesic distances of selected points
        """
        working_radius = radius
        for attempt in range(attempts):
            try:
                euclidean_vertices = self.get_euclidean_patch(vertex, working_radius, old_version=old_version)
                vertex_mask = euclidean_vertices['vertex_mask']
                if vertex_mask.sum() <= 1:
                    working_radius *= 1.1
                    continue
                subsurface = self.create_subsurface(vertex_mask=vertex_mask)
                vertex_map = subsurface.subsurface_vertex_map

                if np.isscalar(vertex):
                    vertex = [vertex]

                geodesic_distance = subsurface.geodesic_distance(vertex_map[vertex], m=m)
                break

            except RuntimeError:
                # singular subsurface
                working_radius *= 1.1
                continue

        else:
            raise Exception('could not find suitable radius')

        close_enough = geodesic_distance <= radius
        close_enough = subsurface.lift_subsurface_data(close_enough)
        geodesic_distance = subsurface.lift_subsurface_data(geodesic_distance) 
        geodesic_distance[~close_enough] = np.nan
        
        vertex_mask = self.get_connected_vertices(vertex=vertex, mask=close_enough, old_version=old_version)

        return {
            'vertex_mask': vertex_mask,
            'geodesic_distance': geodesic_distance[vertex_mask],
        }

    def get_geodesic_patches(self, radius, seeds=None, n_random_seeds=None, output='dense'):
        """create patches of cortex centered around each vertex seed

        - must specify seeds or n_random_seeds

        Parameters
        ----------
        - radius : number
            radius of searchlights
        - seeds : list of ints
            centers of each patch
        - n_random_seeds : int
            number of vertex seeds to generate
        - output : 'dense' or 'sparse'
            'dense': output as dense binary array (faster, less memory efficient)
            'sparse': output as sparse binary array (slower, more memory efficient)
        """

        # gather seeds
        if n_random_seeds is not None:
            seeds = np.random.choice(self.pts.shape[0], n_random_seeds, replace=False)
        if seeds is None:
            raise Exception('must specify seeds or n_random_seeds')

        # initialize output
        output_dims = (len(seeds), self.pts.shape[0])
        if output == 'dense':
            patches = np.zeros(output_dims, dtype=bool)
        elif output == 'sparse':
            patches = scipy.sparse.dok_matrix(output_dims, dtype=bool)
        else:
            raise Exception('output: ' + str(output))

        # compute patches
        for vs, vertex_seed in enumerate(seeds):
            patch = self.get_geodesic_patch(radius=radius, vertex=vertex_seed)
            patches[vs, :] = patch['vertex_mask']

        return {
            'vertex_masks': patches,
        }

    def lift_subsurface_data(self, data, vertex_mask=None):
        """expand vertex dimension of data to original surface's size

        - agnostic to dtype and dimensionality of data
            - vertex dimension should be last dimension

        Parameters
        ----------
        - data : array
            data to lift into original surface dimension
        - vertex_mask : boolean array
            custom mask to use instead of subsurface_vertex_mask
        """
        if vertex_mask is None:
            vertex_mask = self.subsurface_vertex_mask

        new_shape = [vertex_mask.shape[0]]
        if data.ndim > 1:
            new_shape = list(data.shape[:-1]) + new_shape
        lifted = np.zeros(new_shape, dtype=data.dtype)
        lifted[..., vertex_mask] = data

        return lifted

    def get_geodesic_strip_patch(self, v0, v1, radius, room_factor=2, method='bb',
                                 graph_search='astar', include_strip_coordinates=True):
        """return patch that includes v0, v1, their geodesic path, and all points within some radius

        Algorithms
        ----------
        - selection algorithms:
            - 'bb' = big ball
                1. use euclidean ball big enough to contain v0 and v1
                    - center = (v0 + v1) / 2
                    - radius = euclidean_distance(v0, v1) / 2
                2. only proceed if geodesic path [v0 -> v1] does not touch boundary
                    - otherwise expand ball and try again
                3. go along each point in geodesic path, taking geodesic ball of radius r
            - 'graph_distance' = get graph shortest graph path from v0 to v1
                1. take eucidean tube around graph path
                2. will want to use weighted graph instead of just graph
                - this is the fastest method, but requires further implementation tuning
            - 'whole_surface' = use entire surface
        - when geodesic touches the boundary
            1. add euclidean ball of boundary point to working set
            2. recompute
        - for now use:
            - 'bb' when creating single strips or small strips
            - 'whole_surface' when creating many large strips

        Parameters
        ----------
        - v0 : int
            index of start point
        - v1 : int
            index of end point
        - radius : number
            radius of section around geodesic path
        - method : str
            algorithm, either 'bb' or 'graph_distance'
        - room_factor : number
            in bb method, how much extra room in big ball
        - graph_search : 'astar' or 'dijkstra'
            graph search method to use
        - include_strip_coordinates : bool
            whether to compute coordinates of strip
        """

        # find initial submesh that contains v0, v1, and their geodesic path
        if method == 'bb':
            # use a big ball centered between v0 and v1
            xyz_0 = self.pts[v0]
            xyz_1 = self.pts[v1]
            bb_center = (xyz_0 + xyz_1) / 2.0
            bb_radius = room_factor * (((xyz_0 - xyz_1) ** 2).sum() ** 0.5)
            bb = self.get_euclidean_ball(xyz=bb_center, radius=bb_radius)
            initial_mask = self.get_connected_vertices(vertex=v0, mask=bb)
            initial_mask += self.get_connected_vertices(vertex=v1, mask=bb)
            initial_surface = self.create_subsurface(vertex_mask=initial_mask)

            geodesic_path = initial_surface.geodesic_path(
                a=initial_surface.subsurface_vertex_map[v0],
                b=initial_surface.subsurface_vertex_map[v1],
            )

            # collect points within radius of each point in geodesic path
            strip_mask = self.get_geodesic_patch(
                vertex=np.where(initial_surface.subsurface_vertex_mask)[0][geodesic_path],
                radius=radius,
            )

        elif method == 'graph_distance':

            raise NotImplementedError()

            # # use shortest path between v0 and v1 along graph edges
            # import networkx

            # graph = self.weighted_distance_graph
            # if graph_search == 'dijkstra':
            #     graph_path = networkx.shortest_path(graph, v0, v1, weight='weight')
            # elif graph_search == 'astar':
            #     graph_path = networkx.shortest_paths.astar.astar_path(graph, v0, v1, weight='weight')
            # else:
            #     raise Exception(str(graph_search))

            # initial_vertices = self.get_euclidean_patch(
            #     vertex=graph_path,
            #     radius=(radius * room_factor),
            # )
            # initial_mask = initial_vertices['vertex_mask']
            # initial_surface = self.create_subsurface(vertex_mask=initial_mask)

        elif method == 'whole_surface':

            initial_surface = self
            geodesic_path = self.geodesic_path(v0, v1)
            strip_mask = self.get_geodesic_patch(
                vertex=geodesic_path,
                radius=radius,
            )

        else:
            raise Exception('method: ' + str(method))

        geodesic_path_mask = np.zeros(initial_surface.pts.shape[0], dtype=bool)
        geodesic_path_mask[geodesic_path] = True

        # verify geodesic path does not touch boundary
        if (geodesic_path_mask * initial_surface.boundary_vertices).sum() > 2:
            raise Exception('irregular submesh, geodesic path touches boundary')

        output = {
            'vertex_mask': strip_mask['vertex_mask'],
            'geodesic_path': geodesic_path,
        }

        if include_strip_coordinates:
            subsurface = self.create_subsurface(vertex_mask=strip_mask['vertex_mask'])
            coordinates = subsurface.get_strip_coordinates(
                v0=subsurface.subsurface_vertex_map[v0],
                v1=subsurface.subsurface_vertex_map[v1],
                geodesic_path=subsurface.subsurface_vertex_map[geodesic_path],
            )
            output['subsurface'] = subsurface
            output['coordinates'] = subsurface.lift_subsurface_data(coordinates['coordinates'])

        return output

    def get_strip_coordinates(self, v0, v1, geodesic_path=None, distance_algorithm='softmax'):
        """get 2D coordinates of surface from v0 to v1

        - first coordinate: distance along geodesic path from v0
        - second coordinate: distance from geodesic path
        - v0 and v1 should be on boundary of patch
            - if not, they are reassigned to boundary_vertices
        - could be optimized by
            - reusing information from get_geodesic_strip_patch()

        Parameters
        ----------
        - v0 : int
            index of starting point
        - v1 : int
            index of starting point
        - geodesic_path : list of int
            geodesic_path to use
        - distance_algorithm : str
            method to use for computing distance along path, 'softmax' or 'closest'
        """
        if geodesic_path is None:
            geodesic_path = self.geodesic_path(v0, v1)

        geodesic_distances = np.vstack([self.geodesic_distance([v]) for v in geodesic_path])
        v0_distance = geodesic_distances[0, :]

        bound = self.boundary_vertices

        # reassign v0 and v1 to border vertices
        # find boundary vertex maximizing distance to 2nd point in geodesic path
        # s.t. (distance to second point) - (distance to first point) > 0
        if not bound[v0]:

            # use boundary vertex v that minimizes [ d(geopath[0], v) - d(geopath[1], v) ] & > 0
            candidates = bound * (geodesic_distances[0, :] < geodesic_distances[1, :])
            if candidates.sum() == 0:
                bound_max = np.argmax(
                    geodesic_distances[1, bound]
                    - geodesic_distances[0, bound]
                )
                candidates = np.zeros(self.pts.shape[0], dtype=bool)
                candidates[bound[bound_max]] = True

            index = np.argmax(geodesic_distances[1, :][candidates])
            new_v0 = np.where(candidates)[0][index]
            new_path_0 = self.geodesic_path(new_v0, v0)[:-1]
            new_geodesic_distances_0 = np.vstack([self.geodesic_distance([v]) for v in new_path_0])

            v0 = new_v0
            geodesic_path = np.hstack([new_path_0, geodesic_path])
            geodesic_distances = np.vstack([new_geodesic_distances_0, geodesic_distances])

        if not bound[v1]:

            # use boundary vertex v that minimizes [ d(geopath[-1], v) - d(geopath[-2], v) ] & > 0
            candidates = bound * (geodesic_distances[-1, :] < geodesic_distances[-2, :])
            if candidates.sum() == 0:
                bound_max = np.argmax(
                    geodesic_distances[-2, bound]
                    - geodesic_distances[-1, bound]
                )
                candidates = np.zeros(self.pts.shape[0], dtype=bool)
                candidates[bound[bound_max]] = True

            index = np.argmax(geodesic_distances[1, :][candidates])
            new_v1 = np.where(candidates)[0][index]
            new_path_1 = self.geodesic_path(v1, new_v1)[1:]
            new_geodesic_distances_1 = np.vstack([self.geodesic_distance([v]) for v in new_path_1])

            v1 = new_v1
            geodesic_path = np.hstack([geodesic_path, new_path_1])
            geodesic_distances = np.vstack([geodesic_distances, new_geodesic_distances_1])

        # compute distance along line
        if distance_algorithm == 'softmax':
            path_distances = geodesic_distances[0, geodesic_path]
            exp = np.exp(-geodesic_distances)
            softmax = (exp / exp.sum(0))
            distance_along_line = softmax.T.dot(path_distances)
        elif distance_algorithm == 'closest':
            closest_path_vertex = np.array(geodesic_path)[np.argmin(geodesic_distances, axis=0)]
            distance_along_line = v0_distance[closest_path_vertex]
        else:
            raise Exception(distance_algorithm)

        # compute distance from line
        # Calling directly self.geodesic_distance(geodesic_path) is somehow
        # not precise enough on patches, probably because we don't deal
        # correctly with boundaries in the heat method solver. Here instead,
        # we call self.geodesic_distance on each point and take the min.
        distance_from_line = np.min([self.geodesic_distance([ii]) for ii in geodesic_path], axis=0)
        
        # compute the sign for each side of the line
        geodesic_mask = np.zeros(self.pts.shape[0], dtype=bool)
        geodesic_mask[geodesic_path] = True
        subsurface = self.create_subsurface(vertex_mask=(~geodesic_mask))
        whole_submask = np.ones(subsurface.pts.shape[0], dtype=bool)
        connected_component = subsurface.get_connected_vertices(vertex=0, mask=whole_submask)
        subsubmask = np.where(subsurface.subsurface_vertex_mask)[0][connected_component]
        distance_from_line[subsubmask] *= -1

        return {
            'geodesic_path': geodesic_path,
            'coordinates': np.vstack([distance_along_line, distance_from_line]),
            'v0': v0,
            'v1': v1,
        }

    @property
    def furthest_border_points(self):
        """return pair of points on surface border that have largest pairwise geodesic distance"""

        border_mask = self.boundary_vertices
        border_vertices = np.nonzero(border_mask)[0]
        n_border_vertices = border_vertices.shape[0]
        border_pairwise_distances = np.zeros((n_border_vertices, n_border_vertices))
        for v, vertex in enumerate(border_vertices):
            border_pairwise_distances[v, :] = self.geodesic_distance([vertex])[border_mask]
        max_index = np.argmax(border_pairwise_distances)
        v0, v1 = np.unravel_index(max_index, border_pairwise_distances.shape)
        return {'v0': border_vertices[v0], 'v1': border_vertices[v1]}

    def plot_subsurface_rotating_gif(
        self, path, N_frames=48, fps=12, angles=None, vis_data=None,
        disp_patch_verticies=False, disp_patch_edges=False,
        disp_patch_triangles=True, disp_subpatch=False, disp_rim_points=True,
        disp_rim_edges=True, point_color='b', line_color='k', face_color='r'
    ):
        """create a rotating gif of subsurface

        - matplotlib has extremely limited support for 3d plotting
            - expect graphical artifacts when combining multiple features
            - e.g. plotted vertices are not properly obscured by plotted faces
        """

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import mpl_toolkits.mplot3d as a3

        if angles is None:
            elev = 10 * np.ones((N_frames,))
            azim = np.linspace(0, 360, N_frames, endpoint=False)
            angles = list(zip(elev, azim))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('off')

        def init():
            ax.view_init(elev=angles[0][0], azim=angles[0][1])

            if vis_data is not None:
                for plot in vis_data:
                    defaults = {
                        'color': 'c',
                        'marker': '.',
                        'markersize': 15,
                        'linestyle': '',
                    }
                    defaults.update(plot['kwargs'])
                    ax.plot(
                        self.pts[plot['mask'], 0],
                        self.pts[plot['mask'], 1],
                        self.pts[plot['mask'], 2],
                        **defaults
                    )

                tri_poly = a3.art3d.Poly3DCollection(
                    [self.ppts[p, :, :] for p in range(self.ppts.shape[0])],
                    # facecolor='none',
                    alpha=1.0,
                    linewidths=1,
                )
                tri_poly.set_facecolor('red')
                tri_poly.set_edgecolor('black')
                ax.add_collection3d(tri_poly)

            else:
                if True:
                    alpha = 1 if disp_patch_verticies else 0
                    ax.plot(
                        self.pts[:, 0],
                        self.pts[:, 1],
                        self.pts[:, 2],
                        (point_color + '.'),
                        markersize=15,
                        alpha=alpha,
                    )

                if disp_patch_edges:
                    pass

                if disp_patch_triangles:
                    tri_poly = a3.art3d.Poly3DCollection(
                        [self.ppts[p, :, :] for p in range(self.ppts.shape[0])],
                        alpha=1.0,
                    )
                    tri_poly.set_color(face_color)
                    tri_poly.set_edgecolor(line_color)
                    ax.add_collection3d(tri_poly)

        def animate(i):
            ax.view_init(elev=angles[i][0], azim=angles[i][1])
            return []

        anim = animation.FuncAnimation(
            fig,
            animate,
            N_frames,
            interval=25,
            blit=False,
            init_func=init,
        )
        anim.save(path, writer='imagemagick', fps=fps)
