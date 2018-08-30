"""
todo:
- correctness
    - make sure ccw
    - test on sphere/cube/tetrahedron mesh
- features
    - goodrows attr
- optimizations
    - laplace_operator_abbrev ( dont need extra terms B and D?)
    - see if complex64 provides a speedup without loss of accuracy
    - can make vertex ordering faster by storing ordering as array of size (n_v x max_neighbors)
        - fill in blank spaces with nans
- sanitation
    - clean up namespace
        - don't use general words that other files might want to use (e.g. "phi")
        - rename things to be intuitive
    - use connected instead of v_neighbors
    - non vector diffusion
        - is_surface_closed to Surface
- changes to surface
- intrinsic delaunay implementation
    https://github.com/alecjacobson/gptoolbox/blob/master/mesh/intrinsic_delaunay_cotmatrix.m


toodo next:
get the pt_phi computations in order
    - make sur ethe normalization is proper
        - might ned to multiply by 2pi
    - reorder properly


things to check
- is the proper laplacian matrix obtained from self.laplace_operator
- make sure orderings are clockwise
- delunay triangulation
- is r_ij conjugate to r_ji

at beginning of geodesic_distance:
    scipy.sparse.linalg.use_solver(useUmfpack=(self.npts.shape[0] > 10000))




questions for authors
- page 39:7
- (14) on , is beta assumed to be negative and thats why it has the opposite form of alpha
- below (16), what should the subscript on the phi be?


# References
- logarithmic / exponential map
    - Schmidt 2006. Interactive Decal Compositing with Discrete Exponential Maps.
        http://www.unknownroad.com/publications/ExpMapSIGGRAPH06.pdf
    - Melvaer 2012. Geodesic Polar Coordinates on Polygonal Meshes.
        http://heim.ifi.uio.no/~eivindlm/dgpc/DGPC.pdf
    - Sharp 2018. The Vector Heat Method.
        https://arxiv.org/pdf/1805.09170.pdf
- geodesic distance
    - Crane 2013. Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
        https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paperTOG.pdf
"""


import numpy as np
import scipy.sparse

from . import misc


class NewSurfaceProperties(object):
    """eventually just put these into surface"""

    @property
    @misc._memo
    def cotangent_weights(self):

        s0 = self.ppts[:, 1, :] - self.ppts[:, 2, :]
        s1 = self.ppts[:, 2, :] - self.ppts[:, 0, :]
        s2 = self.ppts[:, 0, :] - self.ppts[:, 1, :]

        s0n = np.linalg.norm(s0, axis=1)
        s1n = np.linalg.norm(s1, axis=1)
        s2n = np.linalg.norm(s2, axis=1)

        a2 = -(s0 * s1).sum(axis=1)
        a1 = -(s0 * s2).sum(axis=1)
        a0 = -(s1 * s2).sum(axis=1)

        a2 /= s0n
        a1 /= s0n
        a0 /= s1n

        a2 /= s1n
        a1 /= s2n
        a0 /= s2n

        a = np.vstack([a0, a1, a2]).T

        cotangent_weights = a / np.sqrt(1 - a ** 2)

        cotangent_weights[np.isinf(cotangent_weights)] = 0
        cotangent_weights[np.isnan(cotangent_weights)] = 0

        return cotangent_weights

    @property
    @misc._memo
    def v_neighbors(self):
        v_neighbors = []
        adj = self.adj
        for i in range(self.pts.shape[0]):
            v_neighbors.append((adj.indices[adj.indptr[i]:adj.indptr[i + 1]]))
        v_neighbors = np.array(v_neighbors)
        return v_neighbors

    @property
    @misc._memo
    def faces_of_vertices(self):
        n_polys = self.polys.shape[0]
        values = np.zeros(n_polys * 3)
        values[:n_polys] = 0
        values[n_polys: (2 * n_polys)] = 1
        values[(2 * n_polys): (3 * n_polys)] = 2

        n_pts = self.pts.shape[0]
        n_polys = self.polys.shape[0]

        polys_vs = np.hstack([self.polys[:, 0], self.polys[:, 1], self.polys[:, 2]])
        polys_is = np.hstack([np.arange(n_polys), np.arange(n_polys), np.arange(n_polys)])
        faces_of_vertices = scipy.sparse.coo_matrix(
            (values, (polys_vs, polys_is)),
            (n_pts, n_polys),
        )
        faces_of_vertices = faces_of_vertices.tocsr()

        return faces_of_vertices

    @property
    @misc._memo
    def poly_edge_lengths(self):

        return np.vstack(
            [
                ((self.pts[self.polys[:, 1]] - self.pts[self.polys[:, 2]]) ** 2).sum(1) ** 0.5,
                ((self.pts[self.polys[:, 2]] - self.pts[self.polys[:, 0]]) ** 2).sum(1) ** 0.5,
                ((self.pts[self.polys[:, 0]] - self.pts[self.polys[:, 1]]) ** 2).sum(1) ** 0.5,
            ]
        ).T

    @property
    @misc._memo
    def poly_edge_angles(self):

        el = self.poly_edge_lengths
        el2 = el ** 2

        cos_angles = np.vstack(
            [
                (el2[:, 1] + el2[:, 2] - el2[:, 0]) / (2 * el[:, 1] * el[:, 2]),
                (el2[:, 2] + el2[:, 0] - el2[:, 1]) / (2 * el[:, 2] * el[:, 0]),
                (el2[:, 0] + el2[:, 1] - el2[:, 2]) / (2 * el[:, 0] * el[:, 1]),
            ]
        ).T

        return np.arccos(cos_angles)

    @property
    @misc._memo
    def is_surface_closed(self):
        """return True if surface is closed, checked via Euler's formula"""

        n_vertices = self.pts.shape[0]
        n_edges = self.adj.data.shape[0] / 2.0
        n_polys = self.polys.shape[0]

        # T = 2V - 4
        euler_1 = (n_polys) == (2 * n_vertices - 4)

        # 2E = 3T
        euler_2 = (2 * n_edges) == (3 * n_polys)

        return euler_1 and euler_2


class LogMapMixin(object):
    """implementation of log map, computed using parallel transport"""

    def log_map(self, v0, phi0=0, coordinates='polar'):
        """compute logarithmic map about a given point and initial angle

        ## TODO
        - be able to choose specific vertex for orientation of phi0
        """

        # compute radius
        r = self.geodesic_distance(v0)

        # compute angle
        H = self.parallel_transport(vectors=self.log_map_h_vectors(v0, phi0))
        R = self.parallel_transport(vectors=self.log_map_r_vectors(v0))
        phi = R - H

        # convert coordinates
        if coordinates == 'polar':
            return np.vstack([r, phi])
        elif coordinates == 'cartesian':
            return np.vstack([r * np.cos(phi), r * np.sin(phi)])
        else:
            raise Exception('unknown coordinates: ' + str(coordinates))

    def log_map_h_vectors(self, v0, phi0=0):
        """simply a unit vector emanating from v0

        todo: be able to choose phi0 more intuitively
        """

        vectors = np.zeros(self.pts.shape[0], dtype=complex)  # make sparse
        vectors[v0] = np.exp(1j * phi0)
        return vectors

    def log_map_r_vectors(self, v0):
        """vector integral of circle vector radiating from v0

        (see appendix A of [Sharp 2018])
        """

        v0_polys_sparse = self.faces_of_vertices[v0, :]
        v0_polys_indices = v0_polys_sparse.indices
        v0_poly_edge_lengths = self.poly_edge_lengths[v0_polys_indices]
        v0_poly_edge_angles = self.poly_edge_angles[v0_polys_indices]
        v0_polys = self.polys[v0_polys_indices]

        x = np.zeros(self.pts.shape[0], dtype=complex)

        # neighbors
        neighbors = v0_polys[v0_polys != v0]
        lengths = v0_poly_edge_lengths[v0_polys != v0]
        angles = v0_poly_edge_angles[v0_polys == v0]
        f_angles = angles * np.sin(angles) + 1j * (np.sin(angles) - angles * np.cos(angles))
        areas = self.face_areas[v0_polys_indices]
        x[neighbors[0::2]] = (lengths[0::2] * f_angles / areas)[:, np.newaxis]
        x[neighbors[1::2]] += (lengths[1::2] * f_angles / areas)[:, np.newaxis]
        x[neighbors] *= np.exp(self.pt_phi[v0, :].data)

        # origin
        v0_ordering = self.vertices_neighbor_order[v0]
        v0_polys_sets = [set(poly) for poly in v0_polys]
        v0_polys_sets_ordered = [
            {v0, v1, v2}
            for v1, v2 in zip(v0_ordering, v0_ordering[1:] + v0_ordering[0:1])
        ]
        v0_polys_order = [v0_polys_sets.index(poly) for poly in v0_polys_sets_ordered]
        v0_polys_ordered = v0_polys[v0_polys_order]
        v0_mask = v0_polys_ordered == v0
        v1_mask = v0_polys_ordered == np.array(v0_ordering)[:, np.newaxis]
        areas = self.face_areas[v0_polys_indices][v0_polys_order]
        alpha = v0_poly_edge_angles[v0_polys_order, :][:, v0_mask]
        l_ik = v0_poly_edge_lengths[v0_polys_order, :][:, v1_mask]
        l_ij = np.roll(l_ik, 0)
        x_tilde_ij = 1 / (4 * areas) * (
            -np.sin(alpha) * (l_ik * alpha + l_ij * np.sin(alpha))
            + 1j * (
                l_ij * (np.cos(alpha) * np.sin(alpha) - alpha)
                + l_ik * (alpha * np.cos(alpha) - np.sin(alpha))
            )
        )
        x[v0] = (np.exp(self.pt_phi[v0, :].data) * x_tilde_ij).sum()

        return x


class OrderVerticesMixin(object):
    """methods for ordering vertices according to winding direction"""

    @property
    @misc._memo
    def do_normals_point_outward(self):
        """return True if face normals point outward

        see https://math.stackexchange.com/a/2427372
        """

        # could use x, y, or z
        x_index = 0

        face_midpoints = self.pts[self.polys, :].mean(1)
        div_terms = face_midpoints[:, x_index] * self.face_normals[:, x_index] * self.face_areas
        div_sum = div_terms.sum()
        return div_sum > 0

    @property
    @misc._memo
    def is_polys_winding_direction_consistent(self):
        """return True if winding direction of self.polys is consistent across faces

        - each edge should appear twice, once in each direction
        - each ordered pair of polys neighboring vertices should appear exactly once
        - this check will fail if surface is not closed
        """
        if not self.is_surface_closed:
            raise NotImplementedError('surface not closed')

        edge_dirs = set()
        for poly in self.polys:
            edges = {
                (poly[0], poly[1]),
                (poly[1], poly[2]),
                (poly[2], poly[0]),
            }
            edge_dirs.update(edges)
        return len(edge_dirs) == 3 * self.polys.shape[0]

    @property
    @misc._memo
    def is_polys_winding_direction_ccw(self):
        """return True if winding direction of polys is uniformly ccw"""
        return self.is_polys_winding_direction_consistent and self.do_normals_point_outward

    @property
    @misc._memo
    def ccw_vertices_neighbor_order(self):
        """return ccw ordering of neighbors of each vertex"""
        order = self._vertices_neighbor_order()
        order = self._flip_cw_loops(order)
        return order

    def _vertices_neighbor_order(self):
        """return topological path of neighbors around each vertex

        - each path might be cw or ccw
        - uses graph traversal, optimized with array operations where possible
        - basic sparse strategy:
            - create a 2d array all edges of all polys of each vertex
            - shape = (3 * 3 * n_polys, 2)
            - shape = (3 * 2 * n_edges, 2)
            - shape = (3 * n_polys_of_each_vertex * n_vertices, 2)
            - for each vertex, search through this edge list until path is found
        - cannot use adjacency matrix for this task!
            - includes edges between neighbors that are not contained in polys of origin vertex!
        - TODO: futher parallelize the loop such that all vertex paths searched simultaneously
        - TODO: does accounting for wrapping in edge eliminate all later need for ccw reordering?
        """
        faces_of_vertices = self.faces_of_vertices

        # vertices of polys of each vertex, shape = (3 * n_polys, 3) = (2 * n_edges, 3)
        poly_vs_by_v = self.polys[faces_of_vertices.indices, :]

        # edges of poly_vs_by_v, shape = (3 * 3 * n_polys, 2) = (3 * 2 * n_edges, 2)
        edge_vs_by_v = np.zeros((3 * poly_vs_by_v.shape[0], 2), dtype=int)
        edge_vs_by_v[0::3, :] = poly_vs_by_v[:, 0:2]
        edge_vs_by_v[1::3, :] = poly_vs_by_v[:, 1:3]
        edge_vs_by_v[2::3, :] = poly_vs_by_v[:, [0, 2]]

        # vertex of each row of poly_vs_by_v, shape = (3 * n_polys) = (2 * n_edges)
        poly_vs_by_v_index = np.repeat(
            np.arange(self.pts.shape[0]),
            faces_of_vertices.indptr[1:] - faces_of_vertices.indptr[:-1],
        )

        # vertex of each row of edge_vs_by_v, shape = (3 * 3 * n_polys, 2) = (3 * 2 * n_edges, 2)
        edge_vs_by_v_index = np.zeros((poly_vs_by_v_index.shape[0] * 3, 2))
        edge_vs_by_v_index[0::3, 0] = poly_vs_by_v_index
        edge_vs_by_v_index[1::3, 0] = poly_vs_by_v_index
        edge_vs_by_v_index[2::3, 0] = poly_vs_by_v_index
        edge_vs_by_v_index[0::3, 1] = poly_vs_by_v_index
        edge_vs_by_v_index[1::3, 1] = poly_vs_by_v_index
        edge_vs_by_v_index[2::3, 1] = poly_vs_by_v_index

        # mask of whether each listed edge is opposite to its vertex index
        opposite_edge_mask = (edge_vs_by_v == edge_vs_by_v_index).sum(1) == 0

        # edges opposite to each vertex, shape = (3 * n_polys, 2) = (2 * n_edges, 2)
        opposite_edges = edge_vs_by_v[opposite_edge_mask, :]

        order = []
        for v in np.arange(self.pts.shape[0]):

            # gather edges opposite to vertex
            v_edges = opposite_edges[faces_of_vertices.indptr[v]:faces_of_vertices.indptr[v + 1]]
            n_v_edges = v_edges.shape[0]

            # initialize order with first edge
            v_order = list(v_edges[0, :])

            # unused will keep track of which edges have not yet been added to v_order
            unused = v_edges[1:, :]

            # iterate unused edges until long enough path has been found
            while len(v_order) < n_v_edges:
                next_unused = []

                for e in unused:

                    if len(v_order) == n_v_edges:
                        break

                    # append edge entry to end of v_order if it fits
                    if e[0] == v_order[-1]:
                        v_order.append(e[1])
                    elif e[1] == v_order[-1]:
                        v_order.append(e[0])

                    # prepend edge entry to beginning of v_order if it fits
                    elif e[0] == v_order[0]:
                        v_order.insert(0, e[1])
                    elif e[1] == v_order[0]:
                        v_order.insert(0, e[0])

                    # otherwise edge is unused
                    else:
                        next_unused.append(e)

                unused = next_unused

            order.append(v_order)

        return order

    def _flip_cw_loops(self, vertex_order):
        """modify vertex_order in-place to make ccw ordered"""

        if not self.is_polys_winding_direction_consistent:
            raise NotImplementedError('polys winding direction must be consistent')

        # align vertex neighbor ordering to winding direction
        aligned_to_winding = self._does_vertex_order_match_poly_winding(vertex_order)

        # if polys winding direction is not ccw, reverse targeted alignment
        if self.is_polys_winding_direction_ccw:
            cw = ~aligned_to_winding
        else:
            cw = aligned_to_winding

        # reverse ordering of vertex neighbors to match targeted alignment
        for v in np.nonzero(cw)[0]:
            vertex_order[v] = vertex_order[v][0:1] + vertex_order[v][-1:0:-1]
            # vertex_order[v] = vertex_order[v][::-1]

        return vertex_order

    def _does_vertex_order_match_poly_winding(self, vertex_order):
        """return whether the ordering of each vertex's neighbors matches winding direction"""

        if not self.is_polys_winding_direction_consistent:
            raise NotImplementedError('polys winding direction must be consistent')

        n_pts = self.pts.shape[0]

        # gather a single directed edge opposite to each vertex
        # vertex neighbor ordering will be aligned to this edge

        # take edge from first face of each vertex
        first_face_of_vertex = self.faces_of_vertices.indices[self.faces_of_vertices.indptr[:-1]]

        # collect indices of neighbors of vertices
        neighbor_mask = self.polys[first_face_of_vertex] != np.arange(n_pts)[:, np.newaxis]

        # use the default ordering of mesh since it has consistent winding direction
        directed_edges = self.polys[first_face_of_vertex][neighbor_mask].reshape(-1, 2)

        # for index wrapping, reverse order when indices are (0, 2) instead of (0, 1) or (1, 2)
        reverse_direction = ~neighbor_mask[:, 1]
        directed_edges[reverse_direction, :] = np.roll(directed_edges[reverse_direction, :], 1, 1)

        # check whether vertex_order matches directed edge for each vertex
        order_matches_winding = np.zeros(n_pts, dtype=bool)
        for v0, v_order in enumerate(vertex_order):

            v1, v2 = directed_edges[v0, :]

            # index of first vertex in edge
            i = v_order.index(v1)

            # check whether second vertex in edge follows or precedes first vertex
            # wrap around end of list if need be
            if i == len(v_order) - 1:
                order_matches_winding[v0] = v_order[0] == v2
            else:
                order_matches_winding[v0] = v_order[i + 1] == v2

        return order_matches_winding

    def _is_vertices_neighbor_order_topologically_valid(self, vertex_order):
        bad = {}

        poly_set = set()
        for poly in self.polys:
            poly_set.add(tuple(sorted(poly)))

        for v0, v0_neighbors in enumerate(vertex_order):
            for v1, v2 in zip(v0_neighbors, v0_neighbors[1:] + v0_neighbors[0:1]):
                poly = tuple(sorted([v0, v1, v2]))

                if poly not in poly_set:
                    bad.setdefault(v0, [])
                    bad[v0].append(poly)

        return len(bad) == 0


class ParallelTransportMixin(NewSurfaceProperties, OrderVerticesMixin, LogMapMixin):
    """implementation of parallel transport by Vector Heat method [Sharp 2018]

    - immediate application is global logarithmic map [see log_map()]
    """

    def parallel_transport(self, vectors, m=1.0):
        """solve parallel transport problem via vector heat method

        Properties
        ----------
        - notation
            - P = parallel transport operator
                - P : C^v -> C^v
            - X, Y = vectors
            - rot = rotation operator
        - properties
            - linearity: P(aX + Y) = a P(X) + P(Y)
            - conservation of magnitude: |P(X)| = |X|
            - rotation covariance: P(rot(X)) = rot(P(X))
            - transport symmetry: P(P(delta_i) delta_j)_i = delta_i

        Parameters
        ----------
        - vectors: vector of initial condition vectors to transport
        """

        for attr in ['pt_MtL_solvers', 'pt_MtL_nabla_solvers']:
            if not hasattr(self, attr):
                setattr(self, attr, {})

        # precompute solver
        if m not in self.pt_MtL_solvers or m not in self.pt_MtL_nabla_solvers:

            B, D, W, V = self.laplace_operator
            t = m * self.avg_edge_length ** 2

            # lumped mass matrix
            # M = D
            M = self.pt_lumped_mass_matrix

            # laplacian
            L = V - W

            # connection laplacian
            L_nabla = self.pt_r.copy()
            L_nabla.data *= -W.data
            L_nabla += V

            # poisson equations
            MtL = M + t * L
            MtL_nabla = M + t * L_nabla

            # cache solvers
            self.pt_MtL_solvers[m] = scipy.sparse.linalg.dsolve.factorized(MtL)
            self.pt_MtL_nabla_solvers[m] = scipy.sparse.linalg.dsolve.factorized(MtL_nabla)

        # get solver
        MtL_solve = self.pt_MtL_solvers[m]
        MtL_nabla_solve = self.pt_MtL_nabla_solvers[m]

        # build initial conditions
        Y0 = vectors
        u0 = np.abs(vectors)
        psi0 = np.ones(vectors.shape[0])

        # compute solution
        Y = MtL_nabla_solve(Y0)
        u = MtL_solve(u0)
        psi = MtL_solve(psi0)
        # u_psi = MtL_solve(np.vstack([u0, psi0]).T.copy())
        # u = u_psi[:, 0]
        # psi = u_psi[:, 1]

        return (u * Y) / (psi * np.abs(Y))

    @property
    @misc._memo
    def pt_lumped_mass_matrix(self):
        faces_of_vertices_bool = self.faces_of_vertices.astype(bool)
        faces_of_vertices_bool.data[:] = True
        array = faces_of_vertices_bool.dot(self.face_areas) / 3
        return scipy.sparse.diags(array).tocsr()

    @property
    @misc._memo
    def pt_phi(self):
        """phi for parallel transport algorithm, encodes angles within vertex coordinates"""

        # get ccw ordering of neighbors of each vertex
        n_neighbors = self.adj.indptr[1:] - self.adj.indptr[:-1]
        vertex_order = self.ccw_vertices_neighbor_order
        shifted_order_vertices = [
            v_order[1:] + v_order[0:1]
            for v_order in vertex_order
        ]
        order_vertices_flat = [neighbor for neighbors in vertex_order for neighbor in neighbors]
        shifted_order_vertices_flat = [neighbor for neighbors in shifted_order_vertices for neighbor in neighbors]

        # vertex index of each edge, shape = (2 * n_edges)
        vertices_of_edges = np.repeat(np.arange(self.pts.shape[0]), n_neighbors)

        # origin vertex of each edge
        p1 = self.pts[vertices_of_edges]

        # neighbor vertices
        p2 = self.pts[order_vertices_flat]

        # shifted neighbor vertices
        p3 = self.pts[shifted_order_vertices_flat]

        # direction to each neighbor
        v1 = p1 - p2

        # shifted direction to each neighbor
        v2 = p1 - p3

        # angle of (neighbor -> origin -> next_neighbor), shape = (2 * n_edges)
        cos_angles = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
        angles = np.arccos(cos_angles)

        # # unnormalized angle, used only for quick summing of normalization factors
        # theta = self.adj.copy()

        # cummulative normalized angle
        phi = self.adj.copy()

        n_pts = self.pts.shape[0]
        for v in range(n_pts):
            i1 = self.adj.indptr[v]
            i2 = self.adj.indptr[v + 1]

            # theta.data[i1:i2] = angles[i1:i2]

            # get cumsum of ordered neighbor angles
            angle_cumsum = np.zeros(i2 - i1)
            angle_cumsum[1:] = np.cumsum(angles[(i1 + 1):i2])

            # reorder cumsum based on sorted index ordering
            reordering = np.argsort(vertex_order[v])

            # normalize to 2 pi
            norm_factor = (angle_cumsum[-1] + angles[i1]) / (2 * np.pi)

            # put into phi
            phi.data[i1:i2] = angle_cumsum[reordering] / norm_factor

        return phi

    @property
    @misc._memo
    def pt_r(self):
        """r for parallel transport algorithm, encodes conversion between vertex coordinates"""

        # phi = cummulative normalized angle
        phi = self.pt_phi.tocoo()

        # rho = rotation between neighboring local coordinate systems
        rho = scipy.sparse.coo_matrix(
            (
                np.hstack([-phi.data, phi.data]),
                (np.hstack([phi.row, phi.col]), np.hstack([phi.col, phi.row]))
            ),
            shape=(self.pts.shape[0], self.pts.shape[0]),
        ).tocsr()
        rho.data += np.pi

        # r = complex form of rho
        r = rho.copy()
        r.data = np.exp(1j * rho.data)

        return r

    # def order_vertices_parallel(self):
    #     """

    #     TODO: not done, might need a redesign
    #     """
    #     faces_of_vertices = self.faces_of_vertices

    #     ps_all = self.polys[faces_of_vertices.indices, :]
    #     es_all = np.zeros((3 * ps_all.shape[0], 2))
    #     es_all[0::3, :] = ps_all[:, 0:2]
    #     es_all[1::3, :] = ps_all[:, 1:3]
    #     es_all[2::3, :] = ps_all[:, [0, 2]]

    #     v_of_ps = np.repeat(
    #         np.arange(self.pts.shape[0]),
    #         faces_of_vertices.indptr[1:] - faces_of_vertices.indptr[:-1],
    #     )
    #     v_of_es = np.zeros((v_of_ps.shape[0] * 3, 2))
    #     v_of_es[0::3, 0] = v_of_ps
    #     v_of_es[1::3, 0] = v_of_ps
    #     v_of_es[2::3, 0] = v_of_ps
    #     v_of_es[0::3, 1] = v_of_ps
    #     v_of_es[1::3, 1] = v_of_ps
    #     v_of_es[2::3, 1] = v_of_ps

    #     non_mask = (es_all == v_of_es).sum(1) == 0
    #     # v_of_non = v_of_es[non_mask][:, 0]

    #     non_es_all = es_all[non_mask, :]
    #     non_es_all = non_es_all.astype(int)

    #     n_pts = self.pts.shape[0]
    #     v_neighbors = self.v_neighbors

    #     current_indices = faces_of_vertices.indptr[:-1].copy()

    #     final_paths = np.zeros((n_pts, 20), dtype=int)

    #     path_starts = non_es_all[current_indices][:, 0].copy()
    #     path_ends = non_es_all[current_indices][:, 1].copy()
    #     final_paths[:, 0] = path_starts.copy()
    #     final_paths[:, 1] = path_ends.copy()

    #     finished_indices = np.zeros(non_es_all.shape[0], dtype=bool)
    #     finished_indices[current_indices] = True
    #     current_indices += 1

    #     current_lengths = 2 * np.ones(n_pts, dtype=int)
    #     n_neighbors = np.array([len(q) for q in v_neighbors])

    #     active_indices = np.ones(n_pts, dtype=bool)

    #     loop = 2
    #     while loop < 10:
    #         inner_loop = 0

    #         while inner_loop < 10:

    #             active_e2 = non_es_all[current_indices][:, 0][active_indices]
    #             active_e1 = non_es_all[current_indices][:, 1][active_indices]
    #             add_e2 = path_ends[active_indices] == active_e2
    #             add_e1 = path_ends[active_indices] == active_e1

    #             # need to make sure only one of the 4 paths is actually used

    #             final_paths[active_e2[add_e2], current_lengths[active_indices][add_e2]] = active_e2[add_e2]
    #             final_paths[active_e1[add_e1], current_lengths[active_indices][add_e1]] = active_e1[add_e1]

    #             path_ends[active_indices][add_e2] = active_e2[add_e2]
    #             path_ends[active_indices][add_e1] = active_e1[add_e1]

    #             current_lengths[active_indices][add_e2] += 1
    #             current_lengths[active_indices][add_e1] += 1

    #     #         current_indices[add_e2] += 1
    #     #         current_indices[add_e1] += 1
    #     #         current_indices += 1
    #     #         current_indices += 1
    #             current_indices += 1
    #             overflow = current_indices >= faces_of_vertices.indptr[1:]
    #             current_indices[overflow] = faces_of_vertices.indptr[:-1][overflow]

    #             finished_paths = current_lengths == n_neighbors

    #             active_indices = ~(overflow + finished_paths)

    #             inner_loop += 1

    #         overflow = current_indices >= faces_of_vertices.indptr[1:]
    #         current_indices[overflow] = faces_of_vertices.indptr[:-1][overflow]
    #         active_indices = ~(finished_paths)

    #         loop += 1

    # def order_vertices_geo(self):
    #     """use geometry information to orient vertices

    #     need to solve
    #     - could do this by checking that polys' angle at vertex contributes pos or neg to winding number
    #         - if neg, swap it's vertices in the ordering
    #     """
    #     n_pts = self.pts.shape[0]
    #     adj = self.adj
    #     indptr = adj.indptr
    #     adj_ind = adj.indices
    #     vertex_normals = self.vertex_normals

    #     # obtain direction of each edge
    #     edges = np.vstack(
    #         [
    #             np.repeat(np.arange(n_pts), np.diff(adj.indptr)),
    #             adj.indices,
    #         ]
    #     ).T
    #     edge_dirs = self.pts[edges[:, 0], :] - self.pts[edges[:, 1], :]
    #     edge_dirs /= np.linalg.norm(edge_dirs, axis=1, keepdims=True)

    #     # project edge directions onto vertex normals
    #     edge_vertex_normals = vertex_normals[edges[:, 0]]
    #     projection = (edge_vertex_normals * edge_dirs).sum(axis=1)

    #     # obtain component orthogonal to vertex normal
    #     orthog = edge_dirs - projection[:, np.newaxis] * vertex_normals[edges[:, 0]]
    #     orthog /= np.linalg.norm(orthog, axis=1, keepdims=True)

    #     # obtain angle between each orthogonal component and first neighbor's orthogonal component
    #     first_neighbor_orthog = orthog[adj.indptr[:-1], :][edges[:, 0]]
    #     last_coordinate = np.cross(edge_vertex_normals, first_neighbor_orthog)
    #     x = (orthog * first_neighbor_orthog).sum(1)
    #     y = (orthog * last_coordinate).sum(1)
    #     angles = np.arctan2(y, x)
    #     angles[angles < 0] += 2 * np.pi

    #     # sort by angle to obtain ordering
    #     argsorted_angles = []
    #     for i in range(adj.indptr.shape[0] - 1):
    #         argsorted_angles.append((angles[indptr[i]: indptr[i + 1]]).argsort())

    #     return {
    #         'x': x,
    #         'y': y,
    #         'argsorted_angles': argsorted_angles,
    #     }


# class ParallelTransportTests(object):
#     """functions for testing parallel transport methods"""

#     @staticmethod
#     def check_vertices_ordering(self, argsorted_angles):
#         # confirm that every subsequent pair of neighbors forms a triangle with v0
#         # the only exception is if the three poinst are on a border

#         badorders = 0

#         poly_set = set()
#         for poly in self.polys:
#             poly_set.add(tuple(sorted(poly)))

#         v_neighbors = self.v_neighbors

#         for v0, argsorted_vertex_sequence in enumerate(argsorted_angles):
#             vertex_sequence = v_neighbors[v0][argsorted_vertex_sequence]
#             for v1, v2 in zip(vertex_sequence, vertex_sequence[1:]):
#                 poly = tuple(sorted([v0, v1, v2]))

#                 subadj = self.adj[v_neighbors[v0], :][:, v_neighbors[v0]].todense()
#                 sum_entries_set = set(np.array(subadj.sum(0)).astype(int).flatten())
#                 if sum_entries_set != {4}:
#                     raise Exception(v0, sum_entries_set)

#                 if poly not in poly_set:
#                     badorders += 1
#                     print(v0, poly)
#                     print(sum_entries_set)
#                     raise Exception(poly)

#             # check last pair too!

#         return True

    # @staticmethod
    # def check_vertex_order(surface, vertex_order):
    #     bad = {}

    #     poly_set = set()
    #     for poly in surface.polys:
    #         poly_set.add(tuple(sorted(poly)))

    #     for v0, v0_neighbors in enumerate(vertex_order):
    #         for v1, v2 in zip(v0_neighbors, v0_neighbors[1:] + v0_neighbors[0:1]):
    #             poly = tuple(sorted([v0, v1, v2]))

    #             if poly not in poly_set:
    #                 bad.setdefault(v0, [])
    #                 bad[v0].append(poly)

    #     return bad

#     @staticmethod
#     def plot_vertices_projections(surface, v0, x, y):
#         import matplotlib.pyplot as plt

#         indptr = surface.adj.indptr
#         v_neighbors = surface.v_neighbors

#         subxs = x[indptr[v0]:indptr[v0 + 1]]
#         subys = y[indptr[v0]:indptr[v0 + 1]]

#         plt.figure(figsize=[10,10])
#         plt.plot(subxs, subys, '.r')
#         for i, (subx, suby) in enumerate(zip(subxs, subys)):
#             plt.text(subx, suby, str(v_neighbors[v0][i]))
#         plt.axis('square')

#     @staticmethod
#     def is_vertex_order_ccw(surface, vertex_order):
#         """check whether vertices ordered ccw around vertex normals"""

#         n_pts = surface.pts.shape[0]
#         n_poly_edges = surface.adj.data.shape[0]

#         origin = np.repeat(np.arange(n_pts), np.diff(surface.adj.indptr))

#         # get direction toward first neighbor
#         first_neighbors = np.repeat([vs[0] for vs in vertex_order], np.diff(surface.adj.indptr))
#         first_neighbor_directions = surface.pts[first_neighbors] - surface.pts[origin]
#         first_neighbor_directions /= np.linalg.norm(first_neighbor_directions, axis=1, keepdims=True)

#         # for every vertex, define a space as follows
#         # - origin = vertex
#         # - dim 1 = vertex normal w.r.t. mesh
#         # - dim 2 = projection of (edge from origin to first neighbor) into plane normal to dim 1
#         # - dim 3 = cross(dim 1, dim 2)
#         dim1 = surface.vertex_normals[origin, :]
#         dim2 = first_neighbor_directions - (first_neighbor_directions * dim1).sum(axis=1, keepdims=True) * dim1
#         dim2 /= np.linalg.norm(dim2, axis=1, keepdims=True)
#         dim3 = np.cross(dim1, dim2)

#         # project edge vectors into vertex's defined space
#         outgoing = np.array([v for vs in vertex_order for v in vs])
#         outgoing_vectors = surface.pts[outgoing, :] - surface.pts[origin, :]
#         proj2 = (dim2 * outgoing_vectors).sum(axis=1, keepdims=True)
#         proj3 = (dim3 * outgoing_vectors).sum(axis=1, keepdims=True)

#         # create mask that will roll edges
#         cycle_mask = np.arange(n_poly_edges) - 1
#         cycle_mask[surface.adj.indptr[:-1]] = surface.adj.indptr[1:] - 1

#         a1_angles = np.arctan2(proj3, proj2)
#         a0_angles = a1_angles[cycle_mask]
#         d_angles = a1_angles - a0_angles
#         # d_angles[d_angles < 0] += 2 * np.pi
#         d_angles[d_angles < -np.pi] += 2 * np.pi
#         d_angles[d_angles > np.pi] -= 2 * np.pi

#         # take sum of angles, ==2pi -> ccw, !=2pi -> cw
#         sum_matrix = surface.adj.copy()
#         sum_matrix.data = d_angles[:, 0]
#         sums = sum_matrix.sum(axis=1)
#         ccw = np.isclose(sums, 2 * np.pi)
#         return {
#             'sums': sums,
#             'ccw': ccw,
#             'proj2': proj2,
#             'proj3': proj3,
#             'a1_angles': a1_angles,
#             'd_angles': d_angles,
#             'dim1': dim1,
#             'dim2': dim2,
#         }
#         # return ccw
