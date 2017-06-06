#!python2
from __future__ import division, print_function
from umi_common import *
from collections import deque
class Distance_matrix:
    '''
        This class it implements a distance matrix. It can be used
        to make a distance transform of the chess board locations.
    '''
    OCCUPIED = -1
    UNREACHABLE = -2
    NOT_FOUND = 1000
    def __init__(self):
        '''
            This method uses the locations of the pieces on the board
            to initialise the distance matrix.
        '''
        self.distance_matrix = [[self.UNREACHABLE for z in range(8)] for x in range(8)]


    def __str__(self):
        '''
            This method prints the distance matrix. The distance
            matrix values are printed to System.out.<p>
            "o" stands for occupied.<p>
            "u" stands for unreachable. High path is necessary there<p>
            any integer value is the distance from the target.
        '''
        output_string = "  12345678\n"
        for x in range(7, -1, -1):
            output_string += chr(ord('a') + x) + " "
            for z in self.distance_matrix[x]:
                if z == self.OCCUPIED:
                    output_string += "o"
                elif z == self.UNREACHABLE:
                    output_string += "u"
                else:
                    output_string += str(z)
            output_string += "\n"
        return(output_string)

    def smallest_positive_neighbour(self, x, z):
        '''
            This method finds the smallest neightbour.
            It arguments are a location on the board. It examines the
            neighbours of that location (assuming 4-connectivity) and
            returns the coordinates of the closest neighbour.
            :param x: The row of the board location.
            :param z: The column of the board location.
            :return: The board coordinates of the closest neighbour. If no
            available neighbour exists None is returned.
        '''
        lowest_value = self.NOT_FOUND
        best_neighbour = None
        # Generate all valid options.
        options = [(u,v) for (u,v) in [(x+1, z), (x-1, z), (x, z+1), (x, z-1)] if (u >=0 and u < 8 and v >=0 and v < 8)]

        for (x_temp, z_temp) in options:
            score = self.distance_matrix[x_temp][z_temp]
            if score < lowest_value and score >= 0:
                lowest_value = score
                best_neighbour = (x_temp, z_temp)
        return (best_neighbour, lowest_value)

    def not_possible(self, target_notation):
        '''
            This method checks if it is not possible to plan a low path
            :param target_notation: The board location to plan a path to.
            :return: True If a low path to that location is not possible.
            False if a low path to that location is possible.
        '''
        (x,z) = to_coordinate(target_notation)
        if self.distance_matrix[x][z] == UNREACHABLE:
            return True
        else:
            return False

    def distance_transform(self, chessboard, target_notation):
        '''
            This method generates the distance transform. It sets the
            correct values in the distance matrix. A call to this method
            is necessary before a call to the smallest_positive_neighbour
            method makes sense. It uses the information stored in the board
            to determine the empty and occupied locations. Then starting
            from the target location (distance 0), it assigns distances
            to the empty neighbours of the target location (distance 1)
            and to their empty neighbours (distance 2) and so on. Until
            the whole board in examined. The empty locations that remain
            at the end of the iteration are unreachable due to obstacles.
            In terms of path planning a high path should be planned then.
            :param target_notation: The target location to use when generating the transform.
        '''
        # Initialize the distance matrix with the current chessboard.
        self.distance_matrix = [[self.UNREACHABLE for z in range(8)] for x in range(8)]
        for notation in chessboard.pieces:
            (x, z) = to_coordinate(notation)
            self.distance_matrix[x][z] = self.OCCUPIED

        # Get the target coordinates.
        (x,z) = to_coordinate(target_notation)

        # Set targetfield to 0 (you want to try move here, after all.)
        self.distance_matrix[x][z] = 0

        options = [(u,v) for (u,v) in [(x+1, z), (x-1, z), (x, z+1), (x, z-1)] if (u >=0 and u < 8 and v >=0 and v < 8)]
        to_process = deque(options)
        while len(to_process) > 0:
            (new_x, new_z) = to_process.popleft()
            if self.distance_matrix[new_x][new_z] == self.UNREACHABLE:
                (best_neighbour, lowest_value) = self.smallest_positive_neighbour(new_x, new_z)
                # If it has no neighbour that has a good score, it is unreachable, so it can remain unchanged.
                if best_neighbour == None:
                    continue
                # Generate all new neighbours, that have a path to the target.
                self.distance_matrix[new_x][new_z] = lowest_value + 1
                new_options = [
                    (u,v) for (u,v) in [(new_x+1, new_z), (new_x-1, new_z), (new_x, new_z+1), (new_x, new_z-1)]
                    if (u >=0 and u < 8 and v >=0 and v < 8 and self.distance_matrix[u][v] == self.UNREACHABLE)
                ]
                to_process.extend(new_options)
