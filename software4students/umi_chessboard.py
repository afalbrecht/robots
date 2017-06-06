#!python2

from __future__ import division, print_function
from visual import *
from visual.graph import *
from visual.controls import *

from umi_common import *

class UMI_chessboard:
    def __init__(self, frameworld, board_size=0.3, position_x_z = (0.15, -0.15), angle_degrees=0):
        # Dimensions of the board
        self.chessboard_size = board_size
        self.field_size = (self.chessboard_size / 8.0)

        # Edges of the locations
        self.wallthck = self.field_size / 15.0
        self.wallhght = self.field_size / 15.0

        # Position of the center of the board
        self.mplhght = (self.chessboard_size / 15.0)
        self.mplcent = self.chessboard_size

        # Colors of the board
        self.board_color_light = (1.0, 1.0, 1.0)
        self.board_color_dark = (1.0, 0.5, 1.0)
        self.beam_color = (0.9, 0.9, 0.9)
        self.beam_color = (0.9, 0.9, 0.9)
        self.black_pieces_color = (0, 0, 0)
        self.white_pieces_color = (1, 1, 1)

        # Set the frame of the chessboard.
        self.framemp = frame(frame=frameworld)
        self.framemp.pos =(0, self.mplhght,0)

        # Heights of the pieces:
        self.pieces_height = {"Pawn" : 0.05, "King" : 0.07, "Rook" : 0.06}
        # Create the board on screen
        self.generate_board()

        # Add the pieces
        self.add_pieces()

        # Set the angle and position of the board, where the rotational axis is H8
        self.set_pos_angle(position_x_z, angle_degrees)


    def remove_piece(self, position):
        '''
        Removes a piece from a stored location on the board, and return the object
        :param position: [a1-h8]
        :return: A VPython object (box/cylinder/pyramid)
        '''
        if position in self.pieces:
            piece_data = self.pieces.pop(position, None)
            return piece_data
        else:
            return None

    def get_board_height(self):
        ''' Gives the height of the board.
            :return: Returns the height of the board in meters.
        '''
        return self.mplhght

    def set_angle_radians(self, radians):
        ''' Sets the angle of the board, based of the corner next to h8
            :param radians: The angle of the board in radians.
        '''
        ## Rotate the board
        self.framemp.axis = (cos(radians),0,sin(radians))
        # Used to read the radians of the board.
        self.board_angle = radians

    def set_angle_degrees(self, degrees):
        ''' Sets the angle of the board, based of the corner next to h8
            :param degrees: The angle of the board in degrees.
        '''
        self.set_angle_radians(radians(degrees))

    def get_angle_radians(self):
        ''' Gives the angle of the board in radians.
            :return: Returns the angle of the board in radians.
        '''
        return self.board_angle

    def get_angle_degrees(self):
        ''' Gives the angle of the board in degrees.
            :return: Returns the angle of the board in degrees.
        '''
        ## Rotate the board
        return degrees(self.get_angle_radians())

    def set_position(self, x, z):
        ''' Sets the horizontal position of the board, based of the corner next to h8
            :param x: The forward distance away from the robot arm
            :param z: The left/right distance away from the robot arm
        '''
        self.framemp.pos.x = x
        self.framemp.pos.z = z

    def get_position(self):
        ''' Returns a copy of the position (so students don't accidentally edit it)
            :param x: The forward distance away from the robot arm
            :return: Tuple containing the x, y and z coordinate.
        '''
        return (self.framemp.pos.x, self.framemp.pos.y, self.framemp.pos.z)

    def set_pos_angle(self, position_x_z, angle_degrees):
        ''' Sets the horizontal position of the board, and afterwards the angle based of the corner next to h8
            :param position_x_z: Tuple in the form (x, z)
            :param angle_degrees: The angle in degrees
        '''
        self.set_position(position_x_z[0], position_x_z[1])
        self.set_angle_degrees(angle_degrees)

    def generate_board(self):
        ''' Generates the visual display of the chessboard.
        '''
        self.mchessboard = box(frame = self.framemp,
                       height = self.mplhght,
                       length = self.chessboard_size,
                       width = self.chessboard_size,
                       pos = (0.5*self.chessboard_size, -0.5*self.mplhght, 0.5*self.chessboard_size),
                       color = self.board_color_light)

        # Draw the beams to create 64 squares
        self.width_beams = []
        self.vert_beams = []
        for field in range(8):
            beam_offset = field * (self.chessboard_size / 8.0)
            self.width_beams.append(box(frame = self.framemp,
                       height = self.wallhght,
                       length = self.wallthck,
                       width = self.mchessboard.width,
                       pos = (beam_offset+(0.5*self.wallthck), 0.5*self.wallhght, 0.5*self.mchessboard.width),
                       color = self.beam_color)
            )
            self.vert_beams.append(box(frame = self.framemp,
                       height = self.wallhght,
                       length = self.mchessboard.length,
                       width = self.wallthck,
                       pos = (0.5*self.mchessboard.length, 0.5*self.wallhght, beam_offset+(0.5*self.wallthck)),
                       color = self.beam_color)
            )
        self.width_beams.append(box(frame = self.framemp,
                       height = self.wallhght,
                       length = self.wallthck,
                       width = self.mchessboard.width,
                       pos = (self.chessboard_size-(0.5*self.wallthck), 0.5*self.wallhght, 0.5*self.mchessboard.width),
                       color = self.beam_color)
        )
        self.vert_beams.append(box(frame = self.framemp,
                       height = self.wallhght,
                       length = self.mchessboard.length,
                       width = self.wallthck,
                       pos = (0.5*self.mchessboard.length, 0.5*self.wallhght, self.chessboard_size-(0.5*self.wallthck)),
                       color = self.beam_color)
        )

        self.fields = []
        for x in range(8):
            for z in range(8):
                if (x + z) % 2 == 0:
                    self.fields.append( box(frame = self.framemp,
                           height = 0.001,
                           length = self.field_size,
                           width = self.field_size,
                           pos = (self.field_size*(x+1) - self.field_size/2.0, 0, (self.field_size*z) + self.field_size/2),
                           color = self.board_color_dark)
                    )

    def add_pieces(self):
        '''
        Adds and registers the pieces on the chessboard.
        '''
        self.pieces = dict()
        for z in range(8):
            for x in [0,1,6,7]:
                if x == 1 or x == 0:
                    color_c = self.black_pieces_color
                    color_n = "Black"
                else:
                    color_c = self.white_pieces_color
                    color_n = "White"
                if x in [1,6]:
                    piece_name = "Pawn"
                    piece = cylinder(frame = self.framemp,
                        axis = (0, self.pieces_height[piece_name], 0),
                        radius = self.field_size*0.35,
                        pos = (self.field_size*(x+1) - self.field_size/2.0, 0, (self.field_size*z) + self.field_size/2),
                        color = color_c)
                    self.pieces[to_notation((7-x, 7-z))] = [piece, piece_name, color_n]
                elif x in [0,7] and z in [0,7]:
                    piece_name = "Rook"
                    piece = box(frame = self.framemp,
                           height = self.pieces_height[piece_name],
                           length = self.field_size*0.7,
                           width = self.field_size*0.7,
                           pos = (self.field_size*(x+1) - self.field_size/2.0, self.pieces_height[piece_name]/2, (self.field_size*z) + self.field_size/2),
                           color = color_c)
                    self.pieces[to_notation((7-x, 7-z))] = [piece, piece_name, color_n]
                elif x in [0,7] and z == 4:
                    piece_name = "King"
                    piece = pyramid(frame = self.framemp,
                           axis = (0,1,0),
                           height = self.field_size*0.7,
                           length = self.pieces_height[piece_name],
                           width = self.field_size*0.7,
                           pos = (self.field_size*(x+1) - self.field_size/2.0, 0, (self.field_size*z) + self.field_size/2),
                           color = color_c)
                    self.pieces[to_notation((7-x, 7-z))] = [piece, piece_name, color_n]
