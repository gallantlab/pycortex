# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import numexpr as ne
from scipy.spatial import distance
from scipy import sparse
import scipy.sparse.linalg

from . import exact_geodesic
from . import subsurface
from .misc import _memo


class Surface(exact_geodesic.ExactGeodesicMixin, subsurface.SubsurfaceMixin):
    """Represents a single cortical hemisphere surface. Can be the white matter surface,
    pial surface, fiducial (mid-cortical) surface, inflated surface, flattened surface,
    etc.

    Implements some useful functions for dealing with functions across surfaces.

    Parameters
    ----------
    pts : 2D ndarray, shape (total_verts, 3)
        Location of each vertex in space (mm). Order is x, y, z.
    polys : 2D ndarray, shape (total_polys, 3)
        Indices of the vertices in each triangle in the surface.
    """
    def __init__(self, pts, polys):
        self.pts = pts.astype(np.double)
        self.polys = polys

        self._cache = dict()
        self._rlfac_solvers = dict()
        self._nLC_solvers = dict()

    @property
    @_memo
    def ppts(self):
        """3D matrix of points in each face: n faces x 3 points per face x 3 coords per point.
        """
        return self.pts[self.polys]
    
    @property
    @_memo
    def connected(self):
        """Sparse matrix of vertex-face associations.
        """
        npt = len(self.pts)
        npoly = len(self.polys)
        return sparse.coo_matrix((np.ones((3*npoly,)), # data
                                  (np.hstack(self.polys.T), # row
                                   np.tile(range(npoly),(1,3)).squeeze())), # col
                                 (npt, npoly)).tocsr() # size
    @property
    @_memo
    def adj(self):
        """Sparse vertex adjacency matrix.
        """
        npt = len(self.pts)
        npoly = len(self.polys)
        adj1 = sparse.coo_matrix((np.ones((npoly,)),
                                  (self.polys[:,0], self.polys[:,1])), (npt,npt))
        adj2 = sparse.coo_matrix((np.ones((npoly,)),
                                  (self.polys[:,0], self.polys[:,2])), (npt,npt))
        adj3 = sparse.coo_matrix((np.ones((npoly,)),
                                  (self.polys[:,1], self.polys[:,2])), (npt,npt))
        alladj = (adj1 + adj2 + adj3).tocsr()
        return alladj + alladj.T
    
    @property
    @_memo
    def face_normals(self):
        """Normal vector for each face.
        """
        # Compute normal vector direction
        nnfnorms = np.cross(self.ppts[:,1] - self.ppts[:,0], 
                            self.ppts[:,2] - self.ppts[:,0])
        # Normalize to norm 1
        nfnorms = nnfnorms / np.sqrt((nnfnorms**2).sum(1))[:,np.newaxis]
        # Ensure that there are no nans (shouldn't be a problem with well-formed surfaces)
        return np.nan_to_num(nfnorms)

    @property
    @_memo
    def vertex_normals(self):
        """Normal vector for each vertex (average of normals for neighboring faces).
        """
        # Average adjacent face normals
        nnvnorms = np.nan_to_num(self.connected.dot(self.face_normals) / self.connected.sum(1)).A
        # Normalize to norm 1
        return nnvnorms / np.sqrt((nnvnorms**2).sum(1))[:,np.newaxis]

    @property
    @_memo
    def face_areas(self):
        """Area of each face.
        """
        # Compute normal vector (length is face area)
        nnfnorms = np.cross(self.ppts[:,1] - self.ppts[:,0], 
                            self.ppts[:,2] - self.ppts[:,0])
        # Compute vector length
        return np.sqrt((nnfnorms**2).sum(-1)) / 2

    @property
    @_memo
    def cotangent_weights(self):
        """Cotangent of angle opposite each vertex in each face.
        """
        ppts = self.ppts
        cots1 = ((ppts[:,1]-ppts[:,0]) *
                 (ppts[:,2]-ppts[:,0])).sum(1) / np.sqrt((np.cross(ppts[:,1]-ppts[:,0],
                                                                   ppts[:,2]-ppts[:,0])**2).sum(1))
        cots2 = ((ppts[:,2]-ppts[:,1]) *
                 (ppts[:,0]-ppts[:,1])).sum(1) / np.sqrt((np.cross(ppts[:,2]-ppts[:,1],
                                                                   ppts[:,0]-ppts[:,1])**2).sum(1))
        cots3 = ((ppts[:,0]-ppts[:,2]) *
                 (ppts[:,1]-ppts[:,2])).sum(1) / np.sqrt((np.cross(ppts[:,0]-ppts[:,2],
                                                                   ppts[:,1]-ppts[:,2])**2).sum(1))

        # Then we have to sanitize everything..
        cots = np.vstack([cots1, cots2, cots3])
        cots[np.isinf(cots)] = 0
        cots[np.isnan(cots)] = 0
        return cots

    @property
    @_memo
    def laplace_operator(self):
        """Laplace-Beltrami operator for this surface. A sparse adjacency matrix with
        edge weights determined by the cotangents of the angles opposite each edge.
        Returns a 4-tuple (B,D,W,V) where D is the 'lumped mass matrix', W is the weighted
        adjacency matrix, and V is a diagonal matrix that normalizes the adjacencies.
        The 'stiffness matrix', A, can be computed as V - W.

        The full LB operator can be computed as D^{-1} (V - W).
        
        B is the finite element method (FEM) 'mass matrix', which replaces D in FEM analyses.
        
        See 'Discrete Laplace-Beltrami operators for shape analysis and segmentation'
        by Reuter et al., 2009 for details.
        """
        ## Lumped mass matrix
        D = self.connected.dot(self.face_areas) / 3.0

        ## Stiffness matrix
        npt = len(self.pts)
        cots1, cots2, cots3 = self.cotangent_weights
        # W is weighted adjacency matrix
        W1 = sparse.coo_matrix((cots1, (self.polys[:,1], self.polys[:,2])), (npt, npt))
        W2 = sparse.coo_matrix((cots2, (self.polys[:,2], self.polys[:,0])), (npt, npt))
        W3 = sparse.coo_matrix((cots3, (self.polys[:,0], self.polys[:,1])), (npt, npt))
        W = (W1 + W1.T + W2 + W2.T + W3 + W3.T).tocsr() / 2.0
        
        # V is sum of each col
        V = sparse.dia_matrix((np.array(W.sum(0)).ravel(),[0]), (npt,npt))
        
        # A is stiffness matrix
        #A = W - V # negative operator -- more useful in practice

        # For FEM:
        Be1 = sparse.coo_matrix((self.face_areas, (self.polys[:,1], self.polys[:,2])), (npt, npt))
        Be2 = sparse.coo_matrix((self.face_areas, (self.polys[:,2], self.polys[:,0])), (npt, npt))
        Be3 = sparse.coo_matrix((self.face_areas, (self.polys[:,0], self.polys[:,1])), (npt, npt))
        Bd = self.connected.dot(self.face_areas) / 6
        dBd = scipy.sparse.dia_matrix((Bd,[0]), (len(D),len(D)))
        B = (Be1 + Be1.T + Be2 + Be2.T + Be3 + Be3.T)/12 + dBd
        return B, D, W, V

    def mean_curvature(self):
        """Compute mean curvature of this surface using the Laplace-Beltrami operator.
        Curvature is computed at each vertex. It's probably pretty noisy, and should
        be smoothed using smooth().

        Negative values of mean curvature mean that the surface is folded inward
        (as in a sulcus), positive values of curvature mean that the surface is
        folded outward (as on a gyrus).

        Returns
        -------
        curv : 1D ndarray, shape (total_verts,)
            The mean curvature at each vertex.
        """
        B,D,W,V = self.laplace_operator
        npt = len(D)
        Dinv = sparse.dia_matrix((D**-1,[0]), (npt,npt)).tocsr() # construct Dinv
        L = Dinv.dot((V-W))
        curv = (L.dot(self.pts) * self.vertex_normals).sum(1)
        return curv

    def smooth(self, scalars, factor=1.0, iterations=1):
        """Smooth vertex-wise function given by `scalars` across the surface using
        mean curvature flow method (see http://brickisland.net/cs177fa12/?p=302).

        Amount of smoothing is controlled by `factor`.

        Parameters
        ----------
        scalars : 1D ndarray, shape (total_verts,)
            A scalar-valued function across the cortex, such as the curvature
            supplied by mean_curvature.
        factor : float, optional
            Amount of smoothing to perform, larger values smooth more.
        iterations : int, optional
            Number of times to repeat smoothing, larger values smooths more.

        Returns
        -------
        smscalars : 1D ndarray, shape (total_verts,)
            Smoothed scalar values.
        """
        if factor == 0.0:
            return scalars
        
        B,D,W,V = self.laplace_operator
        npt = len(D)
        lfac = sparse.dia_matrix((D,[0]), (npt,npt)) - factor * (W-V)
        goodrows = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]
        lfac_solver = sparse.linalg.dsolve.factorized(lfac[goodrows][:,goodrows])
        to_smooth = scalars.copy()
        for _ in range(iterations):
            from_smooth = lfac_solver((D * to_smooth)[goodrows])
            to_smooth[goodrows] = from_smooth
        smscalars = np.zeros(scalars.shape)
        smscalars[goodrows] = from_smooth
        return smscalars
        
    @property
    @_memo
    def avg_edge_length(self):
        """Average length of all edges in the surface.
        """
        adj = self.adj
        tadj = sparse.triu(adj, 1) # only entries above main diagonal, in coo format
        edgelens = np.sqrt(((self.pts[tadj.row] - self.pts[tadj.col])**2).sum(1))
        return edgelens.mean()

    def surface_gradient(self, scalars, at_verts=True):
        """Gradient of a function with values `scalars` at each vertex on the surface.
        If `at_verts`, returns values at each vertex. Otherwise, returns values at each
        face.

        Parameters
        ----------
        scalars : 1D ndarray, shape (total_verts,)
            A scalar-valued function across the cortex.
        at_verts : bool, optional
            If True (default), values will be returned for each vertex. Otherwise,
            values will be returned for each face.

        Returns
        -------
        gradu : 2D ndarray, shape (total_verts,3) or (total_polys,3)
            Contains the x-, y-, and z-axis gradients of the given `scalars` at either
            each vertex (if `at_verts` is True) or each face.
        """
        pu = scalars[self.polys]
        fe12, fe23, fe31 = [f.T for f in self._facenorm_cross_edge]
        pu1, pu2, pu3 = pu.T
        fa = self.face_areas

        # numexpr is much faster than doing this using numpy!
        #gradu = ((fe12.T * pu[:,2] +
        #          fe23.T * pu[:,0] +
        #          fe31.T * pu[:,1]) / (2 * self.face_areas)).T
        gradu = np.nan_to_num(ne.evaluate("(fe12 * pu3 + fe23 * pu1 + fe31 * pu2) / (2 * fa)").T)
        
        if at_verts:
            return (self.connected.dot(gradu).T / self.connected.sum(1).A.squeeze()).T
        return gradu

    def create_biharmonic_solver(self, boundary_verts, clip_D=0.1):
        r"""Set up biharmonic equation with Dirichlet boundary conditions on the cortical
        mesh and precompute Cholesky factorization for solving it. The vertices listed in
        `boundary_verts` are considered part of the boundary, and will not be included in
        the factorization.

        To facilitate Cholesky decomposition (which requires a symmetric matrix), the
        squared Laplace-Beltrami operator is separated into left-hand-side (L2) and
        right-hand-side (Dinv) parts. If we write the L-B operator as the product of
        the stiffness matrix (V-W) and the inverse mass matrix (Dinv), the biharmonic
        problem is as follows (with `u` denoting non-boundary vertices)

        .. math::
            :nowrap:
            
            \begin{eqnarray}
            L^2_{u} \phi &=& -\rho_{u} \\
            \left[ D^{-1} (V-W) D^{-1} (V-W) \right]_{u} \phi &=& -\rho_{u} \\
            \left[ (V-W) D^{-1} (V-W) \right]_{u} \phi &=& -\left[D \rho\right]_{u}
            \end{eqnarray}

        Parameters
        ----------
        boundary_verts : list or ndarray of length V
            Indices of vertices that will be part of the Dirichlet boundary.

        Returns
        -------
        lhs : sparse matrix
            Left side of biharmonic problem, (V-W) D^{-1} (V-W)
        rhs : sparse matrix, dia
            Right side of biharmonic problem, D
        Dinv : sparse matrix, dia
            Inverse mass matrix, D^{-1}
        lhsfac : cholesky Factor object
            Factorized left side, solves biharmonic problem
        notboundary : ndarray, int
            Indices of non-boundary vertices
        """
        try:
            from scikits.sparse.cholmod import cholesky
            factorize = lambda x: cholesky(x).solve_A
        except ImportError:
            factorize = sparse.linalg.dsolve.factorized
            
        B, D, W, V = self.laplace_operator
        npt = len(D)

        g = np.nonzero(D > 0)[0] # Find vertices with non-zero mass
        #g = np.nonzero((L.sum(0) != 0).A.ravel())[0] # Find vertices with non-zero mass
        notboundary = np.setdiff1d(np.arange(npt)[g], boundary_verts) # find non-boundary verts
        D = np.clip(D, clip_D, D.max())

        Dinv = sparse.dia_matrix((D**-1,[0]), (npt,npt)).tocsr() # construct Dinv
        L = Dinv.dot((V-W)) # construct Laplace-Beltrami operator
        
        lhs = (V-W).dot(L) # construct left side, almost squared L-B operator
        #lhsfac = cholesky(lhs[notboundary][:,notboundary]) # factorize
        lhsfac = factorize(lhs[notboundary][:,notboundary]) # factorize
        
        return lhs, D, Dinv, lhsfac, notboundary

    def _create_interp(self, verts, bhsolver=None):
        """Creates interpolator that will interpolate values at the given `verts` using
        biharmonic interpolation.

        Parameters
        ----------
        verts : 1D array-like of ints
            Indices of vertices that will serve as knot points for interpolation.
        bhsolver : (lhs, rhs, Dinv, lhsfac, notboundary), optional
            A 5-tuple representing a biharmonic equation solver. This structure
            is created by create_biharmonic_solver.
        
        Returns
        -------
        _interp : function
            Function that will interpolate a given set of values across the surface.
            The values can be 1D or 2D (number of dimensions by len `verts`). Any
            number of dimensions can be interpolated simultaneously.
        """
        if bhsolver is None:
            lhs, D, Dinv, lhsfac, notb = self.create_biharmonic_solver(verts)
        else:
            lhs, D, Dinv, lhsfac, notb = bhsolver
        
        npt = len(D)
        def _interp(vals):
            """Interpolate function with values `vals` at the knot points."""
            v2 = np.atleast_2d(vals)
            nd,nv = v2.shape
            ij = np.zeros((2,nv*nd))
            ij[0] = np.array(verts)[np.repeat(np.arange(nv), nd)]
            ij[1] = np.tile(np.arange(nd), nv)
            
            r = sparse.csr_matrix((vals.T.ravel(), ij), shape=(npt,nd))
            vr = lhs.dot(r)
            
            #phi = lhsfac.solve_A(-vr.todense()[notb]) # 29.9ms
            #phi = lhsfac.solve_A(-vr[notb]).todense() # 29.3ms
            #phi = lhsfac.solve_A(-vr[notb].todense()) # 28.2ms
            phi = lhsfac(-vr[notb].todense())
            
            tphi = np.zeros((npt,nd))
            tphi[notb] = phi
            tphi[verts] = v2.T
            
            return tphi

        return _interp

    def interp(self, verts, vals):
        """Interpolates a function between N knot points `verts` with the values `vals`.
        `vals` can be a D x N array to interpolate multiple functions with the same
        knot points.

        Using this function directly is unnecessarily expensive if you want to interpolate
        many different values between the same knot points. Instead, you should directly
        create an interpolator function using _create_interp, and then call that function.
        In fact, that's exactly what this function does.

        See create_biharmonic_solver for math details.

        Parameters
        ----------
        verts : 1D array-like of ints
            Indices of vertices that will serve as knot points for interpolation.
        vals : 2D ndarray, shape (dimensions, len(verts))
            Values at the knot points. Can be multidimensional.

        Returns
        -------
        tphi : 2D ndarray, shape (total_verts, dimensions)
            Interpolated value at every vertex on the surface.
        """
        return self._create_interp(verts)(vals)

    @property
    @_memo
    def _facenorm_cross_edge(self):
        ppts = self.ppts
        fnorms = self.face_normals
        fe12 = np.cross(fnorms, ppts[:,1] - ppts[:,0])
        fe23 = np.cross(fnorms, ppts[:,2] - ppts[:,1])
        fe31 = np.cross(fnorms, ppts[:,0] - ppts[:,2])
        return fe12, fe23, fe31

    def approx_geodesic_distance(self, verts, m=0.1):
        """Computes approximate geodesic distance (in mm) from each vertex in 
        the surface to any vertex in the collection `verts`. This approximation
        is computed using Varadhan's formula for geodesic distance based on the
        heat kernel. This is very fast (quite a bit faster than `geodesic_distance`)
        but very inaccurate. Use with care.

        In short, we let heat diffuse across the surface from sources at `verts`,
        and then look at the resulting heat levels in every other vertex to 
        approximate how far they are from the sources. In theory, this should
        be very accurate as the duration of heat diffusion goes to zero. In 
        practice, short duration leads to numerical instability and error.

        Parameters
        ----------
        verts : 1D array-like of ints
            Set of vertices to compute distance from. This function returns the shortest
            distance to any of these vertices from every vertex in the surface.
        m : float, optional
            Scalar on the duration of heat propagation. Default 0.1.

        Returns
        -------
        1D ndarray, shape (total_verts,)
            Approximate geodesic distance (in mm) from each vertex in the 
            surface to the closest vertex in `verts`.
        """
        npt = len(self.pts)
        t = m * self.avg_edge_length ** 2 # time of heat evolution

        if m not in self._rlfac_solvers:
            B, D, W, V = self.laplace_operator
            nLC = W - V # negative laplace matrix
            spD = sparse.dia_matrix((D,[0]), (npt,npt)).tocsr() # lumped mass matrix
            
            lfac = spD - t * nLC # backward Euler matrix

            # Exclude rows with zero weight (these break the sparse LU)
            goodrows = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]
            self._goodrows = goodrows
            self._rlfac_solvers[m] = sparse.linalg.dsolve.factorized(lfac[goodrows][:,goodrows])

        # Solve system to get u, the heat values
        u0 = np.zeros((npt,)) # initial heat values
        u0[verts] = 1.0
        goodu = self._rlfac_solvers[m](u0[self._goodrows])
        u = np.zeros((npt,))
        u[self._goodrows] = goodu

        return -4 * t * np.log(u)

    def geodesic_distance(self, verts, m=1.0, fem=False):
        """Minimum mesh geodesic distance (in mm) from each vertex in surface to any
        vertex in the collection `verts`.

        Geodesic distance is estimated using heat-based method (see 'Geodesics in Heat',
        Crane et al, 2012). Diffusion of heat along the mesh is simulated and then
        used to infer geodesic distance. The duration of the simulation is controlled
        by the parameter `m`. Larger values of `m` will smooth & regularize the distance
        computation. Smaller values of `m` will roughen and will usually increase error
        in the distance computation. The default value of 1.0 is probably pretty good.

        This function caches some data (sparse LU factorizations of the laplace-beltrami
        operator and the weighted adjacency matrix), so it will be much faster on
        subsequent runs.

        The time taken by this function is independent of the number of vertices in verts.

        Parameters
        ----------
        verts : 1D array-like of ints
            Set of vertices to compute distance from. This function returns the shortest
            distance to any of these vertices from every vertex in the surface.
        m : float, optional
            Reverse Euler step length. The optimal value is likely between 0.5 and 1.5.
            Default is 1.0, which should be fine for most cases.
        fem : bool, optional
            Whether to use Finite Element Method lumped mass matrix. Wasn't used in 
            Crane 2012 paper. Doesn't seem to help any.

        Returns
        -------
        1D ndarray, shape (total_verts,)
            Geodesic distance (in mm) from each vertex in the surface to the closest
            vertex in `verts`.
        """
        npt = len(self.pts)
        if m not in self._rlfac_solvers or m not in self._nLC_solvers:
            B, D, W, V = self.laplace_operator
            nLC = W - V # negative laplace matrix
            if not fem:
                spD = sparse.dia_matrix((D,[0]), (npt,npt)).tocsr() # lumped mass matrix
            else:
                spD = B
            
            t = m * self.avg_edge_length ** 2 # time of heat evolution
            lfac = spD - t * nLC # backward Euler matrix

            # Exclude rows with zero weight (these break the sparse LU)
            goodrows = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]
            self._goodrows = goodrows
            self._rlfac_solvers[m] = sparse.linalg.dsolve.factorized(lfac[goodrows][:,goodrows])
            self._nLC_solvers[m] = sparse.linalg.dsolve.factorized(nLC[goodrows][:,goodrows])

        # I. "Integrate the heat flow ̇u = ∆u for some fixed time t"
        # ---------------------------------------------------------

        # Solve system to get u, the heat values
        u0 = np.zeros((npt,)) # initial heat values
        u0[verts] = 1.0
        goodu = self._rlfac_solvers[m](u0[self._goodrows])
        u = np.zeros((npt,))
        u[self._goodrows] = goodu

        # II. "Evaluate the vector field X = − ∇u / |∇u|"
        # -----------------------------------------------

        # Compute grad u at each face
        gradu = self.surface_gradient(u, at_verts=False)
        
        # Compute X (normalized grad u)
        #X = np.nan_to_num((-gradu.T / np.sqrt((gradu**2).sum(1))).T)
        graduT = gradu.T
        gusum = ne.evaluate("sum(gradu ** 2, 1)")
        X = np.nan_to_num(ne.evaluate("-graduT / sqrt(gusum)").T)

        # III. "Solve the Poisson equation ∆φ = ∇·X"
        # ------------------------------------------

        # Compute integrated divergence of X at each vertex
        #x1 = x2 = x3 = np.zeros((X.shape[0],))
        c32, c13, c21 = self._cot_edge
        x1 = 0.5 * (c32 * X).sum(1)
        x2 = 0.5 * (c13 * X).sum(1)
        x3 = 0.5 * (c21 * X).sum(1)
        
        conn1, conn2, conn3 = self._polyconn
        divx = conn1.dot(x1) + conn2.dot(x2) + conn3.dot(x3)

        # Compute phi (distance)
        goodphi = self._nLC_solvers[m](divx[self._goodrows])
        phi = np.zeros((npt,))
        phi[self._goodrows] = goodphi - goodphi.min()

        # Ensure that distance is zero for selected verts
        phi[verts] = 0.0

        return phi

    def geodesic_path(self, a, b, max_len=1000, d=None, **kwargs):
        """Finds the shortest path between two points `a` and `b`.

        This shortest path is based on geodesic distances across the surface.
        The path starts at point `a` and selects the neighbor of `a` in the 
        graph that is closest to `b`. This is done iteratively with the last
        vertex in the path until the last point in the path is `b`.

        Other Parameters in kwargs are passed to the geodesic_distance 
        function to alter how geodesic distances are actually measured

        Parameters
        ----------
        a : int
            Vertex that is the start of the path
        b : int
            Vertex that is the end of the path
        d : array
            array of geodesic distances, will be computed if not provided

        Other Parameters
        ----------------
        max_len : int, optional, default=1000
            Maximum path length before the function quits. Sometimes it can get stuck
            in loops, causing infinite paths.
        m : float, optional
            Reverse Euler step length. The optimal value is likely between 0.5 and 1.5.
            Default is 1.0, which should be fine for most cases.
        fem : bool, optional
            Whether to use Finite Element Method lumped mass matrix. Wasn't used in 
            Crane 2012 paper. Doesn't seem to help any.

        Returns
        -------
        path : list
            List of the vertices in the path from a to b
        """
        path = [a]
        if d is None:
            d = self.geodesic_distance([b], **kwargs)
        while path[-1] != b:
            n = np.array([v for v in self.graph.neighbors(path[-1])])
            path.append(n[d[n].argmin()])
            if len(path) > max_len:
                return path
        return path

    @property
    @_memo
    def _cot_edge(self):
        ppts = self.ppts
        cots1, cots2, cots3 = self.cotangent_weights
        c3 = cots3[:,np.newaxis] * (ppts[:,1] - ppts[:,0])
        c2 = cots2[:,np.newaxis] * (ppts[:,0] - ppts[:,2])
        c1 = cots1[:,np.newaxis] * (ppts[:,2] - ppts[:,1])
        c32 = c3 - c2
        c13 = c1 - c3
        c21 = c2 - c1
        return c32, c13, c21

    @property
    @_memo
    def _polyconn(self):
        npt = len(self.pts)
        npoly = len(self.polys)
        o = np.ones((npoly,))
        c1 = sparse.coo_matrix((o, (self.polys[:,0], range(npoly))), (npt, npoly)).tocsr()
        c2 = sparse.coo_matrix((o, (self.polys[:,1], range(npoly))), (npt, npoly)).tocsr()
        c3 = sparse.coo_matrix((o, (self.polys[:,2], range(npoly))), (npt, npoly)).tocsr()
        return c1, c2, c3

    @property
    @_memo
    def boundary_vertices(self):
        """return mask of boundary vertices

        algorithm: for simple mesh, every edge appears in either 1 or 2 polys
            1 -> border edge
            2 -> non-border edge
        """

        first = np.hstack(
            [
                self.polys[:, 0],
                self.polys[:, 1],
                self.polys[:, 2],
            ]
        )
        second = np.hstack(
            [
                self.polys[:, 1],
                self.polys[:, 2],
                self.polys[:, 0],
            ]
        )
        polygon_edges = np.vstack([first, second])
        polygon_edges = np.vstack([polygon_edges.min(axis=0), polygon_edges.max(axis=0)])

        sort_order = np.lexsort(polygon_edges)
        sorted_edges = polygon_edges[:, sort_order]
        duplicate_mask = (sorted_edges[:, :-1] == sorted_edges[:, 1:]).sum(axis=0) == 2

        nonduplicate_mask = np.ones(sorted_edges.shape[1], dtype=bool)
        nonduplicate_mask[:-1][duplicate_mask] = False
        nonduplicate_mask[1:][duplicate_mask] = False

        border_mask = np.zeros(self.pts.shape[0], dtype=bool)
        border_mask[sorted_edges[:, nonduplicate_mask][0, :]] = True
        border_mask[sorted_edges[:, nonduplicate_mask][1, :]] = True

        return border_mask

    @property
    def iter_surfedges(self):
        for a, b, c in self.polys:
            yield a, b
            yield b, c
            yield a, c

    @property
    def iter_surfedges_weighted(self):
        """iterate through edges

        - same iteration order as self.edge_lengths
            - border edges will be iterated once, non-border edges will be iterated twice
        """
        distances = self.edge_lengths
        n_edges = distances.size / 3

        for i, (a, b, c) in enumerate(self.polys):
            yield a, b, distances[i]
            yield b, c, distances[i + n_edges]
            yield a, c, distances[i + 2 * n_edges]

    @property
    @_memo
    def graph(self):
        """NetworkX undirected graph representing this Surface.
        """
        import networkx as nx
        graph = nx.Graph()
        graph.add_edges_from(self.iter_surfedges)
        return graph

    def get_graph(self):
        return self.graph

    @property
    @_memo
    def edge_lengths(self):
        """return vector of edge lengths

        - same iteration order as iter_surfedges_listed()
            - border edges will be iterated once, non-border edges will be iterated twice
        """

        n_edges = self.polys.shape[0]
        edges = np.zeros((n_edges * 3, 3))
        edges[:n_edges, :] = self.ppts[:, 0, :] - self.ppts[:, 1, :]
        edges[n_edges:(2 * n_edges), :] = self.ppts[:, 1, :] - self.ppts[:, 2, :]
        edges[(2 * n_edges):, :] = self.ppts[:, 2, :] - self.ppts[:, 0, :]

        edges **= 2
        distances = edges.sum(axis=1)
        distances **= 0.5

        return distances

    @property
    @_memo
    def weighted_distance_graph(self):
        import networkx as nx
        weighted_graph = nx.Graph()
        weighted_graph.add_weighted_edges_from(self.iter_surfedges_weighted)
        return weighted_graph

    def extract_chunk(self, nfaces=100, seed=None, auxpts=None):
        '''Extract a chunk of the surface using breadth first search, for testing purposes'''
        node = seed
        if seed is None:
            node = np.random.randint(len(self.pts))

        ptmap = dict()
        queue = [node]
        faces = set()
        visited = set([node])
        while len(faces) < nfaces and len(queue) > 0:
            node = queue.pop(0)
            for face in self.connected[node].indices:
                if face not in faces:
                    faces.add(face)
                    for pt in self.polys[face]:
                        if pt not in visited:
                            visited.add(pt)
                            queue.append(pt)

        pts, aux, polys = [], [], []
        for face in faces:
            for pt in self.polys[face]:
                if pt not in ptmap:
                    ptmap[pt] = len(pts)
                    pts.append(self.pts[pt])
                    if auxpts is not None:
                        aux.append(auxpts[pt])
            polys.append([ptmap[p] for p in self.polys[face]])

        if auxpts is not None:
            return np.array(pts), np.array(aux), np.array(polys)

        return np.array(pts), np.array(polys)

    def extract_geodesic_chunk(self, origin, radius):
        """Extract a chunk of the surface that is within radius of the origin by
        geodesic distance.
        """
        dist = self.geodesic_distance([origin])
        sel = np.nonzero(dist < radius)[0]
        sel_pts = self.pts[sel]

        # create new polys with remapped indices

        # find polys where all 3 verts are in the selected set
        sel_polys_inds = np.nonzero(self.connected[sel].sum(0) == 3)[1]
        sel_polys_old = self.polys[sel_polys_inds]

        # create array to remap indices in polys to new indices
        keyarr = np.zeros(len(self.pts), dtype=int)
        keyarr[sel] = range(len(sel))

        sel_polys = keyarr[sel_polys_old]

        return sel_pts, sel_polys


    def polyhedra(self, wm):
        '''Iterates through the polyhedra that make up the closest volume to a certain vertex'''
        for p, facerow in enumerate(self.connected):
            faces = facerow.indices
            pts, polys = _ptset(), _quadset()
            if len(faces) > 0:
                poly = np.roll(self.polys[faces[0]], -np.nonzero(self.polys[faces[0]] == p)[0][0])
                assert pts[wm[p]] == 0
                assert pts[self.pts[p]] == 1
                pts[wm[poly[[0, 1]]].mean(0)]
                pts[self.pts[poly[[0, 1]]].mean(0)]

                for face in faces:
                    poly = np.roll(self.polys[face], -np.nonzero(self.polys[face] == p)[0][0])
                    a = pts[wm[poly].mean(0)]
                    b = pts[self.pts[poly].mean(0)]
                    c = pts[wm[poly[[0, 2]]].mean(0)]
                    d = pts[self.pts[poly[[0, 2]]].mean(0)]
                    e = pts[wm[poly[[0, 1]]].mean(0)]
                    f = pts[self.pts[poly[[0, 1]]].mean(0)]

                    polys((0, c, a, e))
                    polys((1, f, b, d))
                    polys((1, d, c, 0))
                    polys((1, 0, e, f))
                    polys((f, e, a, b))
                    polys((d, b, a, c))

            yield pts.points, np.array(list(polys.triangles))

    def patches(self, auxpts=None, n=1):
        def align_polys(p, polys):
            x, y = np.nonzero(polys == p)
            y = np.vstack([y, (y+1)%3, (y+2)%3]).T
            return polys[np.tile(x, [3, 1]).T, y]

        def half_edge_align(p, pts, polys):
            poly = align_polys(p, polys)
            mid   = pts[poly].mean(1)
            left  = pts[poly[:,[0,2]]].mean(1)
            right = pts[poly[:,[0,1]]].mean(1)
            s1 = np.array(np.broadcast_arrays(pts[p], mid, left)).swapaxes(0,1)
            s2 = np.array(np.broadcast_arrays(pts[p], mid, right)).swapaxes(0,1)
            return np.vstack([s1, s2])

        def half_edge(p, pts, polys):
            poly = align_polys(p, polys)
            mid   = pts[poly].mean(1)
            left  = pts[poly[:,[0,2]]].mean(1)
            right = pts[poly[:,[0,1]]].mean(1)
            stack = np.vstack([mid, left, right, pts[p]])
            return stack[(distance.cdist(stack, stack) == 0).sum(0) == 1]

        for p, facerow in enumerate(self.connected):
            faces = facerow.indices
            if len(faces) > 0:
                if n == 1:
                    if auxpts is not None:
                        pidx = np.unique(self.polys[faces])
                        yield np.vstack([self.pts[pidx], auxpts[pidx]])
                    else:
                        yield self.pts[self.polys[faces]]
                elif n == 0.5:
                    if auxpts is not None:
                        pts = half_edge(p, self.pts, self.polys[faces])
                        aux = half_edge(p, auxpts, self.polys[faces])
                        yield np.vstack([pts, aux])
                    else:
                        yield half_edge_align(p, self.pts, self.polys[faces])
                else:
                    raise ValueError
            else:
                yield None

    def edge_collapse(self, p1, p2, target):
        raise NotImplementedError
        face1 = self.connected[p1]
        face2 = self.connected[p2]


class _ptset(object):
    def __init__(self):
        self.idx = OrderedDict()
    def __getitem__(self, idx):
        idx = tuple(idx)
        if idx not in self.idx:
            self.idx[idx] = len(self.idx)
        return self.idx[idx]

    @property
    def points(self):
        return np.array(list(self.idx.keys()))

class _quadset(object):
    def __init__(self):
        self.polys = dict()

    def __call__(self, quad):
        idx = tuple(sorted(quad))
        if idx in self.polys:
            del self.polys[idx]
        else:
            self.polys[idx] = quad

    @property
    def triangles(self):
        for quad in list(self.polys.values()):
            yield quad[:3]
            yield [quad[0], quad[2], quad[3]]
