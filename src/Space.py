from typing import List


class Space(object):

    def __init__(self):
        self.functions = []
        self.depth = 0

    def get_children_of_function(self, level: int, func_index: int) -> List[int]:
        """
        Given the index of a function at the given level, return the indices of the functions on level level+1 corresponding
        to the given function.
        :param level:
        :param func_index:
        :return:
        """

        assert level <= self.depth - 1

        pass

    def get_parent_of_function(self, level: int, func_index: int) -> int:
        """
        Given the index of a function on the given level, return the index of its parent function at the previous level.
        :param level: 
        :param func_index: 
        :return:
        """

        assert 1 <= level

        for func in self.functions[level - 1]:
            if func_index in self.get_children_of_function(level - 1, func):
                return func
        return -1
