from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def visualize_image(image: np.ndarray):
    assert len(image.shape) == 2, "Image must be a 2D array"
    plt.imshow(image, cmap='gray')
    plt.show()

def average_image(images: np.ndarray):
    assert len(images.shape) == 3, "Images must be a 3D array"
    return np.mean(images, axis=0)

def maximize_likelihood(psi: np.ndarray) -> Tuple[np.ndarray, list]:
    """Function to maximize the likelihood of the parameters s and eta given the data psi.

    Args:
        psi (np.ndarray): Average image of the data represented as a vector.
    Returns:
        Tuple[np.ndarray, list]: Returns the shape of the object represented as a vector and the parameters eta.
    """

    psi = psi.flatten()

    s = np.random.randint(2, size=psi.shape[0])

    # Two random numbers between 0 and 1 that sum to 1
    eta = [0.45, 0.55]

    while True:
        s_old = s
        eta_old = eta

        print("Number of ones in s: ", np.sum(s == 1))
        print("Number of zeros in s: ", np.sum(s == 0))
        s = update_s(psi, eta)
        eta = update_eta(psi, s)

        if np.array_equal(s, s_old) and np.array_equal(eta, eta_old):
            break

    s = s.reshape([100, 100])

    return s, eta

def update_s(psi: np.ndarray, eta: list) -> np.ndarray:
    """Function to update the shape of the object.

    Args:
        psi (np.ndarray): Average image of the data represented as a vector.
        eta (list): Two element list of the parameters eta_0 and eta_1.

    Returns:
        np.ndarray: Updated shape of the object represented as a vector.
    """

    assert len(psi.shape) == 1, "Psi must be a vector"
    assert len(eta) == 2, "Eta must be a list of length 2"

    num_pixels = psi.shape[0]
    num_etas = len(eta)

    scores = np.zeros([num_etas, num_pixels])
    scores[0, :] = eta[0]*psi - np.log(1 + np.exp(eta[0]))
    scores[1, :] = eta[1]*psi - np.log(1 + np.exp(eta[1]))

    s = np.argmax(scores, axis=0)

    return s

def update_eta(psi: np.ndarray, s: np.ndarray) -> list:
    """Function to update the parameters eta.

    Args:
        psi (np.ndarray): Average image of the data represented as a vector.
        s (np.ndarray): Shape - binary image of the object represented as a vector.

    Returns:
        list: Updated parameters eta.
    """

    assert len(psi.shape) == 1, "Psi must be a vector"
    assert len(s.shape) == 1, "S must be a vector"

    eta = [0, 0]

    num_zeros = np.sum(s == 0)
    nun_ones = np.sum(s == 1)

    eta[0] = np.log(np.dot(psi, 1 - s) / (num_zeros - np.dot(psi, 1 - s)))
    eta[1] = np.log(np.dot(psi, s) / (nun_ones - np.dot(psi, s)))

    # Eta has to be between 0 and 1
    eta[0] = max(0, eta[0])
    eta[1] = max(0, eta[1])

    return eta



