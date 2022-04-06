import numpy as np
from scipy.spatial.transform import Rotation as R


def mod_contrast(im, mod=0.4):
    """Modify image contrast
    Parameters
    ----------
    im : ndarray
        2D image
    mod : float, optional
        exponent to apply to the image, by default 0.4

    Returns
    -------
    ndarray
        The contrast modified image

    Todo
    ----

    More can definitely be done here to get closer to tag contrast, perhaps using
    a more custom contrast curve, or maybe even some form of masking

    """

    im = im ** mod

    return im


def sim_gridtag(im, spacing=5, total_flip=70, flip_pattern=(1, 3, 3, 1), x_offset=0, y_offset=0):
    """Adds grid tags (line tags in perpendicular directions) to an image

    Parameters
    ----------
    im : ndarray
        2D image that will have grid tag lines applied
    spacing : float
        The distance between lines, in pixels
    total_flip : float, optional
        The total RF flip (degrees) of the tagging pulse, by default 70
        A higher number here, up to 90, will make the dark regions of the tag lines
        darker, with 90 being complete nulling.
    flip_pattern : tuple or ndarray, optional
        The type of tagging pulse to play, by default (1, 3, 3, 1)
        Splitting the tagging pulse into a binomial pulse makes the tag lines
        sharper, other options would be (1, 1) which would give broad lines, and
        (1, 4, 6, 4, 1) for even sharper lines
    x_offset : float, optional
        Offsets the line pattern in the x direction, in pixels, by default 0
    y_offset : int, optional
        Offsets the line pattern in the y direction, in pixels, by default 0

    Returns
    -------
    ndarray
        The input image with grid tag lines added

    Todo
    ----

    Maybe mask out air pixels from getting tag lines?

    """

    flip_pattern = np.array(flip_pattern)

    # Flip angles in radians -- reduce to have less
    flip_angles = total_flip * flip_pattern / flip_pattern.sum() * np.pi / 180

    # We now make a map that represents the spatial gradients that get played out
    # between RF pulses, which determines tag line spacing
    xn, yn = tuple(im.shape[:2])
    X, Y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')
    
    # Offset the origin so lines can be shifted
    X += x_offset
    Y += y_offset

    # Scale factor to have line distance = "spacing"
    moment = (2 * np.pi) / (spacing * np.sqrt(2))
    # Note: if this wasnt diagonal it would be 2 pi / spacing

    # Now we apply the line tages in both perpendicular and diagonal  directions
    G_theta = moment * (X + Y)
    im2 = sim_linetag(im, flip_angles, G_theta)
    
    G_theta = moment * (X - Y)
    im3 = sim_linetag(im2, flip_angles, G_theta)

    return im3


def sim_linetag(im, flip_angles, G_theta):
    """Adds tag lines to an image using a psuedo-Bloch simulation

    Parameters
    ----------
    im : ndarray
        2D image that will have tag lines applied
    flip_angles : ndarray
        1D array of flip angles (radians) for the simulation
    G_theta : ndarray
        2D map of z-axis rotation (radians) for each point in image

    Returns
    -------
    ndarray
        The input image with tag lines added

    Todo
    ----
    There are probably too many vector copies here (Mr, Mr2, Mr3), but the
    wasted memory isnt big...

    Check G_theta is the same size as im
    """
    
    # Magnetization vector for each point in the image
    M = np.zeros((im.size, 3))
    M[:, 2] = im.ravel()
    
    G_theta = G_theta.ravel()

    for i in range(flip_angles.size - 1):
        
        # Tip all spins along the x-axis
        r = R.from_euler('x', flip_angles[i])
        Mr = r.apply(M)

        # Rotate around the z-axis, based on position
        Mr2 = Mr.copy()
        Mr2[:, 0] = np.cos(G_theta) * Mr[:, 0] - np.sin(G_theta) * Mr[:, 1]
        Mr2[:, 1] = np.sin(G_theta) * Mr[:, 0] + np.cos(G_theta) * Mr[:, 1]

        M = Mr2.copy()

    # Apply the final RF tip on the x-axis
    r = R.from_euler('x', flip_angles[i + 1])
    Mr3 = r.apply(Mr2)

    # The final image is whatever magnetization is in the z direction
    im2 = np.abs(Mr3[:, 2])
    im2 = np.reshape(im2, im.shape)

    return im2
