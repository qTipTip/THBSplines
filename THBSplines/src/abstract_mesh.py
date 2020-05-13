from abc import ABC, abstractmethod

import numpy as np


class Mesh(ABC):

    @abstractmethod
    def plot_cells(self) -> None:
        """Visualizes the mesh, if the spatial dimension of the ambient space allows it."""
        pass

    @abstractmethod
    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        """
        Computes gauss quadrature points and weights for the given list of cells.
        :param cell_indices: indices of cells to compute quadrature points for.
        """
        pass
