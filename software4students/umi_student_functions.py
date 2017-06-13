#!python2

from __future__ import division, print_function
from umi_parameters import UMI_parameters
from umi_common import *
import math
import numpy as np
from visual import *
# Specifications of UMI
# Enter the correct details in the corresponding file (umi_parameters.py).
# <<<<<<<<<<-------------------------------------------------------------------- TODO FOR STUDENTS
UMI = UMI_parameters()

def apply_inverse_kinematics(x, y, z, gripper):
    ''' Computes the angles, given some real world coordinates
        :param float x: cartesian x-coordinate
        :param float y: cartesian y-coordinate
        :param float z: cartesian z-coordinate

        :return: Returns the a tuple containing the position and angles of the robot-arm joints.
    '''
    # Implementation is based on the Robotics readers made by Leo.
    # TIP: If you want to know at all times, what the current x,y,z of your robot-arm is,
    # print(get_gripper_bottom_position())
    # Read the other TIP at the bottom of the umi_simulation file.
    
    # Real arm runs from of 0 to 1.082
    riser_position = y + UMI.total_arm_height # (we want the gripper to be at the y position, but we can only influence the riser.)

    # Compute the resulting angles for each joint in DEGREES (you can use the degrees() function to convert radians).
    elbow_angle = 0 # ????
    shoulder_angle = 0 # ????
    # We want the piece to be placed down in the same angle as we picked it up
    wrist_angle = 0 # ????
    # Gripper is not influenced by the kinematics, so one less variable for you to alter *yay*
    return (riser_position, shoulder_angle, elbow_angle, wrist_angle, gripper)

def board_position_to_cartesian(chessboard, position):
    ''' Convert a position between [a1-h8] to its cartesian coordinates in frameworld coordinates.

        You are not allowed to use the functions such as: frame_to_world.
        You have to show actual calculations using positions/vectors and angles.

        :param obj chessboard: The instantiation of the chessboard that you wish to use.
        :param str position: A position in the range [a1-h8]

        :return: tuple Return a position in the format (x,y,z)
    '''
    # Get the local coordinates for the tiles on the board in the 0-7 range.
    (row, column) = to_coordinate(position)
    world_coordinate_x = (row * chessboard.field_size) - 0.5 * chessboard.
    # h8 is closes to the rotation point, row a[1-8] is furthest away from the robot arm.

    # ????? Perform the calculations, also take a rotation and translation of the chessboard in account!

    # Output the results.
    result = (world_coordinate_x, world_coordinate_y, world_coordinate_z)

    return result

def find_piece_height(chessboard, position):
    piece_name = chessboard.pieces[position][1]
    if piece_name == "Pawn":
        piece_height = 0.05
    elif piece_name == "Rook":
        piece_height = 0.06
    elif piece_name == "King":
        piece_height = 0.07
    return piece_height/2

def high_path(chessboard, from_pos, to_pos):
    '''
    Computes the high path that the arm can take to move a piece from one place on the board to another.
    :param chessboard: Chessboard object
    :param from_pos: [a1-h8]
    :param to_pos: [a1-h8]
    :return: Returns a list of instructions for the GUI.
    '''
    sequence_list = []
    # We assume that 20 centimeter above the board is safe.
    safe_height = 0.2
    # We assume that 10 centimeter above the board is "low".
    low_height = 0.1

    # Define half_piece height (you want to grab the middle of a piece, so get the height of the piece on a position.)
    # (*cough* this data might be stored in a chessboard *cough*)
    # You might need if statements around this, but you have to fill this variable regardlessly.
    half_piece_height = find_piece_height(chessboard, from_pos) # ????  ##------------------------------------------------------------<---<--AANPASSEN------------------------------------------------------------------##

    # Get the coordinates.
    (from_x, from_y, from_z) = board_position_to_cartesian(chessboard, from_pos)
    (to_x, to_y, to_z) = board_position_to_cartesian(chessboard, to_pos)

    REPLACE_THIS_WITH_YOUR_OWN_CODE = "wrong"
    # Hover above the first field on SAFE height:
    sequence_list.append(apply_inverse_kinematics(from_x, REPLACE_THIS_WITH_YOUR_OWN_CODE, from_z, chessboard.field_size))
    # Hover above the first field on LOW height:

    # Hover above the first field on half of the piece height:

    # Hover above the first field on half of the piece height:

    # Grip the piece

    # Give instruction to GUI to pickup piece
    sequence_list.append(["GUI", "TAKE", from_pos])
    # Hover above the first field on SAFE height (Keep the gripper closed!!):

    # Move to new position on SAFE height

    # Hover above the first field on LOW height:

    # Hover above the first field on half of the piece height:

    # Give instruction to GUI to drop piece
    sequence_list.append(["GUI", "DROP", to_pos])
    # Move to new position on SAFE height (And open the gripper)

    return sequence_list

def move_to_garbage(chessboard, from_pos):
    '''
        Computes the high path that the arm can take to move a piece from one place on the board to the garbage location.
        :param chessboard: Chessboard object
        :param from_pos: [a1-h8]
        :return: Returns a list of instructions for the GUI.
    '''
    sequence_list = []
    # We assume that 20 centimeter above the board is safe.
    safe_height = 0.2
    # We assume that 10 centimeter above the board is "low".
    low_height = 0.1
    drop_location = "j5"
    # Define half_piece height (you want to grab the middle of a piece, so get the height of the piece on a position.)
    # (*cough* this data might be stored in a chessboard *cough*)
    REPLACE_THIS_WITH_YOUR_OWN_CODE = "wrong"
    # You might need if statements around this, but you have to fill this variable regardlessly.
    half_piece_height = 0 # ????

    # Get the coordinates.
    (from_x, from_y, from_z) = board_position_to_cartesian(chessboard, from_pos)
    (to_x, to_y, to_z) = board_position_to_cartesian(chessboard, drop_location)


    # Hover above the first field on SAFE height:
    sequence_list.append(apply_inverse_kinematics(from_x, REPLACE_THIS_WITH_YOUR_OWN_CODE, from_z, chessboard.field_size))
    # Hover above the first field on LOW height:

    # Hover above the first field on half of the piece height:

    # Hover above the first field on half of the piece height:

    # Grip the piece

    # Give instruction to GUI to pickup piece
    sequence_list.append(["GUI", "TAKE", from_pos])
    # Hover above the first field on SAFE height (Keep the gripper closed!!):

    # Move to new position on SAFE height

    # Hover above the first field on LOW height:

    # Hover above the first field on half of the piece height:

    # Give instruction to GUI to drop piece
    sequence_list.append(["GUI", "DROP", drop_location])
    # Move to new position on SAFE height (And open the gripper)

    return sequence_list
