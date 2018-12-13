from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Space(ABC):

    @abstractmethod
    def cell_to_basis(self, cell_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Given a list of N cell-indices, return a list of N lists, such that the i'th list gives the indices of the functions
        supported over cell i.
        :param cell_indices:
        :return:
        """
        pass

    @abstractmethod
    def basis_to_cell(self, basis_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Given a list of N basis-indices, return a list of N lists, such that the i'th list gives the indices of the cells
        supporting over basis i.
        :param basis_indices:
        :return:
        """
        pass