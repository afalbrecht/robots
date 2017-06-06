#!python2
from __future__ import division, print_function
import csv

def to_coordinate(notation):
    """ Given a notation in the form [a1-h8], return the corresponding notation
        (0-7, 0-7)
        :param str notation: Location of a field on the board

        :return: Tuple internal coordinates of the field.
    """
    z = ord(notation[0]) - ord('a')
    x = int(notation[1]) - 1
    return (x, z)

def to_notation(coordinates):
    """ Given a board coordinate in the form (0-7, 0-7), return the corresponding notation
        [a1-h8]
        :param tuple coordinates: Tuple containing the internal coordinates on the board.

        :return: String in the form 'a1'
    """
    (x,z) = coordinates
    letter = chr(ord('a') + z)
    number = x + 1
    return letter + str(number)

def write_parameters_to_file(parameter_lines, output_file):
    """ Given a list of instructions, save it to a file so it can be read later.
        :param parameter_lines: List containing the intructions to both the arm and the GUI.
        :param output_file: Name of the file in which the output will be stored.
    """
    with open(output_file, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['Riser', 'Shoulder', 'Elbow', 'Wrist', 'Gripper'])
        for line in parameter_lines:
            csv_writer.writerow(line)

def write_parameters_to_umi_robot(parameter_lines):
    """ Given a list of instructions, save it to a file so it can be read later by the actual UMI robot.
        :param parameter_lines: TList containing the intructions to both the arm and the GUI.
    """
    with open("joints.txt", 'wb') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        for line in parameter_lines:
            if len(line) == 5:
                csv_writer.writerow([line[0]*1000.0, line[1], line[2], line[3], -90.0, 0.0, 0.0, line[4]*1000.0])

def read_parameters_from_file(input_file):
    """ Read the file as written by write_parameters_to_file
        :param input_file: Name of the file in which the input is stored.
    """
    with open(input_file, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)
        parameter_lines = []
        for line in csv_reader:
            if len(line) > 3:
                parameter_lines.append([float(x) for x in line])
            elif line[0] == "GUI":
                parameter_lines.append(line)
    return (headers, parameter_lines)
