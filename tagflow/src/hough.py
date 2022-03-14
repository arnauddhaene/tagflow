from typing import Tuple

import cv2
import numpy as np

from ..utils import generate_reference


def hough_circle(
    image: np.ndarray,
    dp: float, min_d: float, p1: float, p2: float,
    min_r: float, max_r: float, circ: int, radial: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    circles = cv2.HoughCircles(image.astype(np.uint8), cv2.HOUGH_GRADIENT,
                               dp=dp, minDist=min_d, param1=p1, param2=p2,
                               minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        circles = np.uint16(np.around(circles))
    else:
        # If no circles found, go for center
        circles = np.array(
            [[[image.shape[0] / 2, image.shape[1] / 2, 25]]])

    Cx, Cy, R = tuple(circles[0, 0])

    r0 = generate_reference((R * .5, R * .9), circ, radial)

    return r0 + np.array([Cx, Cy]), circles[0, 0]
