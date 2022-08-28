import numpy as np
import scipy.linalg

"""
def example_usage():
    # This is not a proper function, just an example of how this code may be used
    

    # --- Define some parameters for RBF interpolation
    rbf_const = 12
    rbf_reg = 1e-3

    # --- Define the analysis points

    # Here we just get the location of every pixel in a mask (idx_mask)
    QP_x = idx_mask[:, 0]
    QP_y = idx_mask[:, 1]

    # Then we subtract the center of these points to center at (0,0)
    cx = QP_x.mean()
    cy = QP_y.mean()
    QP = np.array([QP_x - cx, QP_y - cy])

    # --- Prep tracked points
    
    # rr_in are our tracked points, shape = (N, 2, Nt)
    rr = np.array([rr_in[:, 0, :] - cx, rr_in[:, 1, :] - cy])

    Nt = rr.shape[2]
    Np = QP.shape[1]

    # --- Prepate the RBF interpolation
    rr0 = rr[:, :, 0]
    rbf = RBF(rr0, const=rbf_const, reg=rbf_reg)

    # --- Calcualte strain for each timepoint
    E_res = []
    for t in range(Nt):
        rrt = rr[:, :, t]
        F = np.zeros([Np, 3, 3])
        out = []
        for i in range(2):
            coeff_r = rbf.solve(rrt[i, :])
            F[:, i, :2] = rbf.derivative(QP).T
        E_res.append(get_principle_strain(QP, F))
    E_res = np.array(E_res)

    # E_res now has the princile strain components for each timeframe, and each QP point
"""


def get_principle_strain(QP, F):
    """Returns principle strain components from analysis points and deformation gradients

    Parameters
    ----------
    QP : array-like, shape = (2, N)
        Points to measure strain at.  The principle directions will be computed around (0,0)
    F : array-like, shape = (N, 3, 3)
        Deformation tensor for each QP point

    Returns
    -------
    3 arrays, shape = (N,)
        Ecc, Err, and Ell strains for each of the QP points
    """

    Nqp = QP.shape[1]
    Ecc = np.zeros(Nqp)
    Err = np.zeros(Nqp)
    Ell = np.zeros(Nqp)

    for i in range(Nqp):
        Fq = F[i]
        Fq[2, 2] = 1.0 / (Fq[0, 0] * Fq[1, 1] - Fq[1, 0] * Fq[0, 1])

        unit_vec = QP[:, i] / np.linalg.norm(QP[:, i])

        circ = np.array([unit_vec[1], - unit_vec[0], 0.0])
        radial = np.array([unit_vec[0], unit_vec[1], 0.0])
        longitude = np.array([0.0, 0.0, 1.0])

        # Compute the Green-Lagrangian strain tensor
        E = 0.5 * ((Fq.T @ Fq) - np.eye(3))

        # Get the component E in each principle direction
        Ecc[i] = circ.T @ E @ circ
        Err[i] = radial.T @ E @ radial
        Ell[i] = longitude.T @ E @ longitude
    
    return Ecc, Err, Ell


# Gaussian RBF
def bfunc_gaussian(rad, const):
    return np.exp(-0.5 * rad * rad / (const * const))


# Derivative of Gaussian RBF
def deriv_bfunc_gaussian(rad, const):
    return -rad * np.exp(-0.5 * rad * rad / (const * const)) / (const * const)


class RBF:
    """
    RBF interpolation class for calcualting deformation tensors from tracked points
    """
    
    def __init__(self, nodes, func_type='gaussian', const=16, reg=1e-3):
        """[summary]

        Parameters
        ----------
        nodes : array-like, shape = (2, N)
            The location (x,y) of the N tracked points in the reference configuration
        func_type : str, optional
            The RBF to use, right now only 'gaussian' is supported
        const : float, optional
            The shape parameter used for the RBF kernel, by default 16
        reg : flaot, optional
            Tikhonov regularization used in the fitting process, higher values produce
            more smoothing, by default 1e-3
        """

        self.nodes = nodes.copy()
        self.const = const
        self.reg = reg
        
        self.Ndim = nodes.shape[0]
        self.Nextra = self.Ndim + 1
        self.N = nodes.shape[1]
        
        if func_type.lower() == 'gaussian':
            self.bfunc = bfunc_gaussian
            self.deriv_bfunc = deriv_bfunc_gaussian
            
        # this calculates the pairwise euclidean distance between each point
        # in the reference frame
        # equal to sklearn.metrics.pairwise.euclidean_distances but faster ;)
        rad = nodes[:, :, None] - nodes[:, None, :]
        rad = np.linalg.norm(rad, axis=0)
        
        val = self.bfunc(rad, self.const)
        P = np.vstack([np.ones(self.N), nodes])
        Z = np.zeros([self.Nextra, self.Nextra])
        
        self.A = np.block([[val, P.T], [P, Z]])
        self.A += self.reg * np.eye(self.N + self.Nextra)
        
    def solve(self, b):
        """Solves for the optimal weights using least squares given the deformed point locations

        Currently set up to solve in 1D, so must loop through directions

        Parameters
        ----------
        b : array-like, shape = (N)
            The location of the N tracked points in the deformed configuration, typically this
            is the point locations at a certain time point.
        """

        b = np.hstack((b, np.zeros(self.Nextra)))
        self.current_coeff = scipy.linalg.lstsq(self.A, b)[0]
        # self.current_coeff = np.linalg.solve(self.A, b)
        return self.current_coeff.copy()
    
    def interp(self, x, coeff=None):
        """Interpolates input points from the reference configuation to deformed

        This may be used to estimate the motion of untracked points for visualization
        or other purposes (such as calcualting strain with FDM instead of the RBF derivative)

        Currently set up to solve in 1D, so must loop through directions

        Parameters
        ----------
        x : array-like, shape = (N)
            Point locations in the reference configuration
        coeff : array-like, shape = (N_node + Ndim + 1,), optional
            The coefficients calculated from self.solve, if left blank it will use the last
            calcualted coefficients, by default None

        Returns
        -------
        array-like, shape = (N)
            The predicted location of points 'x' in the deformed configuration
        """

        if coeff is None:
            coeff = self.current_coeff
            
        Nout = x.shape[1]

        rad = x[:, None, :] - self.nodes[:, :, None]
        rad = np.linalg.norm(rad, axis=0)
        val = self.bfunc(rad, self.const)
        val = np.vstack((val, np.ones(Nout), x))
        out = coeff @ val
        
        return out
    
    def derivative(self, x, coeff=None):
        """Calculates the derivative of the RBF at the given points

        Parameters
        ----------
        x : array-like, shape = (N)
            Point locations in the reference configuration
        coeff : array-like, shape = (N_node + Ndim + 1,), optional
            The coefficients calculated from self.solve, if left blank it will use the last
            calcualted coefficients, by default None

        Currently set up to solve in 1D, so must loop through directions

        Returns
        -------
        array-like, shape = (N)
            The derivative of the RBF at location of points 'x' in the deformed configuration
            (this is the deformation tensor)
        """

        if coeff is None:
            coeff = self.current_coeff
            
        Nout = x.shape[1]
        out = np.zeros([self.Ndim, Nout])
        
        for k in range(self.Ndim):
            
            # Derivative of radial basis function
            rad = x[:, None, :] - self.nodes[:, :, None]
            rad = np.linalg.norm(rad, axis=0)
            val = self.deriv_bfunc(rad, self.const)
            
            # times (x[k]-node[k])/rad
            dk = x[k, None, :] - self.nodes[k, :, None]
            val *= dk / rad
            
            # Add a row of ones to add the correct linear component
            P = np.zeros([self.Nextra, val.shape[1]])
            P[k + 1] = 1.0
            
            # Combine
            val = np.vstack((val, P))
            
            # All together is coeff * rbf_derivative * (x-node)/r + linear coeff
            # this matrix format calcualtes all rows at once for speed
            out[k] = coeff @ val
            
        return out
