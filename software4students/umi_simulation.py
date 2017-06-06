#!python2

from __future__ import division, print_function
from visual import *
from visual.graph import *
from visual.controls import *
import wx
from copy import deepcopy
# Custom made imports
from umi_parameters import UMI_parameters
from umi_chessboard import UMI_chessboard
from umi_student_functions import *
import numpy as np
import os.path

#**********************************************
# ROBOT PARAMETERS

# Specifications of UMI ARE IMPORTED THROUGH umi_student_functions.

#**********************************************
# Functions that are called on various events

def setRiserHeight(evt): # called on slider events (output in mm)
    value = s0.GetValue() / 1000.0
    moveRiser(value)

def moveRiser(value):
    '''
    Sets the height of the riser, values are entered in meters
    :param value: Value in meters, while the display is in mm.
    '''
    s0_label.SetLabel('Set Riser Height: %d mm' % (value * 1000.0))
    s0.SetValue(value * 1000.0)
    riser.pos.y = UMI.correct_height(value)
    UMI_angles[0] = value


def setShoulderAngle(evt): # called on slider events (output in degrees)
    value = s1.GetValue() / 1000.0
    moveShoulder(value)

def moveShoulder(value):
    '''
    Sets the angle of the shoulder joint, values are entered in radians
    :param value: Angle in radians.
    '''
    s1_label.SetLabel('Set Shoulder rotation: %.2f degrees' % degrees(value))
    shoulder_joint.axis = (cos(value),0,sin(value))
    UMI_angles[1] = value
    s1.SetValue(value*1000.0)

def setElbowAngle(evt): # called on slider events (output in degrees)
    value = s2.GetValue() / 1000.0
    moveElbow(value)

def moveElbow(value):
    '''
    Sets the angle of the elbow joint, values are entered in radians
    :param value: Angle in radians.
    '''
    s2_label.SetLabel('Set Elbow rotation: %.2f degrees' % degrees(value))
    elbow_joint.axis = (cos(value),0,sin(value))
    UMI_angles[2] = value
    s2.SetValue(value*1000.0)

def setWristAngle(evt): # called on slider events] (output in degrees)
    value = s3.GetValue() / 1000.0
    moveWrist(value)

def moveWrist(value):
    '''
    Sets the angle of the wrist joint, values are entered in radians
    :param value: Angle in radians.
    '''
    s3_label.SetLabel('Set Wrist rotation: %.2f degrees' % degrees(value))
    wrist_joint.axis = (cos(value),0,sin(value))
    UMI_angles[3] = value
    s3.SetValue(value*1000.0)

def setGripperWidth(evt): # called on slider events] (output in degrees)
    value = s4.GetValue() / 1000.0
    moveGripper(value)

def moveGripper(value):
    '''
    Sets the distance between the grippers, values are entered in meters
    :param value: Distance between grippers in meters.
    '''
    s4_label.SetLabel('Set Gripper opening: %d mm' % (value * 1000))
    gripper_pos.pos = (0, gripper_pos.pos.y, 0.5*gripper_pos.width+value/2)
    gripper_neg.pos = (0, gripper_pos.pos.y, -0.5*gripper_pos.width-value/2)
    UMI_angles[4] = value
    s4.SetValue(value*1000.0)

def read_input_file(evt):
    """
        Upon clicking the button, reads the input from the file, and executes the instructions.
    """
    joints_file = "joints_simulator.txt"
    if os.path.isfile(joints_file):
        (headers, sequence_list) = read_parameters_from_file(joints_file)
        execute_sequence(sequence_list)

def store_input_text(evt):
    """
        Upon clicking the button, reads the instruction in the textbox, and translates this to instruction for the
        robot arm to execute this action.
    """
    joints_file = "joints_simulator.txt"
    input_text = input_field.GetValue()
    if len(input_text) == 4:
        parameter_lines = move(CHESSBOARD, input_text[0:2], input_text[2:4])
        write_parameters_to_file(parameter_lines, joints_file)

L = 600
# Create a window. Note that w.win is the wxPython "Frame" (the window).
# window.dwidth and window.dheight are the extra width and height of the window
# compared to the display region inside the window. If there is a menu bar,
# there is an additional height taken up, of amount window.menuheight.
# The default style is wx.DEFAULT_FRAME_STYLE; the style specified here
# does not enable resizing, minimizing, or full-sreening of the window.
w = window(width=2*(L+window.dwidth), height=L+window.dheight,
           menus=False, title='UMI RTX',
           style= wx.CAPTION | wx.CLOSE_BOX)

# Place a 3D display widget in the left half of the window.
d = 20
disp = display(window=w, x=d, y=d, width=L-2*d, height=L-2*d, forward=-vector(1,0.25,1), center=vector(0,0.5,0))
disp.background=(0.859, 0.949, 0.957)
# Place buttons, radio buttons, a scrolling text object, and a slider
# in the right half of the window. Positions and sizes are given in
# terms of pixels, and pos(0,0) is the upper left corner of the window.
p = w.panel # Refers to the full region of the window in which to place widgets

wx.StaticText(p, pos=(d,4), size=(L-2*d,d), label='3D representation.',
              style=wx.ALIGN_CENTRE | wx.ST_NO_AUTORESIZE)

s0 = wx.Slider(p, pos=(1.0*L,0.1*L), size=(0.9*L,20), minValue=UMI.joint_ranges["Riser"][0]*1000.0, maxValue=UMI.joint_ranges["Riser"][1]*1000.0)
s0.Bind(wx.EVT_SCROLL, setRiserHeight)
s0_label = wx.StaticText(p, pos=(1.0*L,0.05*L), label='Set Riser height: %d mm' % (UMI.joint_ranges["Riser"][1]*1000.0))

s1 = wx.Slider(p, pos=(1.0*L,0.2*L), size=(0.9*L,20), minValue=radians(UMI.joint_ranges["Shoulder"][0])*1000.0, maxValue=radians(UMI.joint_ranges["Shoulder"][1])*1000.0)
s1.Bind(wx.EVT_SCROLL, setShoulderAngle)
s1_label = wx.StaticText(p, pos=(1.0*L,0.15*L), label='Set Shoulder rotation: 0 degrees')

s2 = wx.Slider(p, pos=(1.0*L,0.3*L), size=(0.9*L,20), minValue=radians(UMI.joint_ranges["Elbow"][0])*1000.0, maxValue=radians(UMI.joint_ranges["Elbow"][1])*1000.0)
s2.Bind(wx.EVT_SCROLL, setElbowAngle)
s2_label = wx.StaticText(p, pos=(1.0*L,0.25*L), label='Set Elbow rotation: 0 degrees')

s3 = wx.Slider(p, pos=(1.0*L,0.4*L), size=(0.9*L,20), minValue=radians(UMI.joint_ranges["Wrist"][0])*1000.0, maxValue=radians(UMI.joint_ranges["Wrist"][1])*1000.0, style=wx.SL_HORIZONTAL)
s3.Bind(wx.EVT_SCROLL, setWristAngle)
s3_label = wx.StaticText(p, pos=(1.0*L,0.35*L), label='Set Wrist rotation: 0 degrees')

s4 = wx.Slider(p, pos=(1.0*L,0.5*L), size=(0.9*L,20), minValue=UMI.joint_ranges["Gripper"][0]*1000.0, maxValue=UMI.joint_ranges["Gripper"][1]*1000.0, style=wx.SL_HORIZONTAL)
s4.Bind(wx.EVT_SCROLL, setGripperWidth)
s4_label = wx.StaticText(p, pos=(1.0*L,0.45*L), label='Set Gripper opening: 50 mm')

read_input = wx.Button(p, label='Execute joints.txt', pos=(1.0*L,0.75*L))
read_input.Bind(wx.EVT_BUTTON, read_input_file)

store_input = wx.Button(p, label='Compute High Path', pos=(1.11*L,0.65*L))
store_input.Bind(wx.EVT_BUTTON, store_input_text)

input_field = wx.TextCtrl(p, pos=(1.0*L,0.65*L), value='a1a3',
            size=(0.1*L,25))
input_field.SetInsertionPoint(len(input_field.GetValue())+1) # position cursor at end of text
input_field.SetFocus() # so that keypresses go to the TextCtrl without clicking it

#***********************************************
# ROBOT JOINTS
frameworld = frame()

frame0 = frame(frame=frameworld)
frame0.pos = (-UMI.wpedestal/2.0, 0.5*UMI.hpedestal,0)

# The shoulder joint location is now on world position (x,z) = (0,0)
riser = frame(frame=frame0)
riser.pos = (UMI.wpedestal/2.0,frame0.pos.y, 0)

shoulder_joint = frame(frame=riser)
shoulder_joint.pos = (0,-UMI.pedestal_offset, 0)
#shoulder_joint.rotate(axis = (0, 1, 0), angle = pi/4)

elbow_joint = frame(frame=shoulder_joint)
elbow_joint.pos = (UMI.upper_length,-UMI.upper_height, 0)
#elbow_joint.rotate(axis = (0, 1, 0), angle = pi/4)

wrist_joint = frame(frame=elbow_joint)
wrist_joint.pos = (UMI.lower_length,-UMI.lower_height, 0)
#wrist_joint.rotate(axis = (0, 1, 0), angle = pi/4)
#************************************************
# ROBOT ARM
pedestal = box(frame = frame0,
               pos = (0,0,0),
               height = UMI.hpedestal,
               length = UMI.wpedestal,
               width = UMI.wpedestal,
               color = (0.4, 0.4, 0.4))
riser_part = cylinder(frame = riser,
               pos = (0, -UMI.pedestal_offset, 0),
               axis = (0, UMI.pedestal_offset, 0),
               radius = UMI.wpedestal/2.0,
               color = color.red)

upper_arm = box(frame = shoulder_joint,
               pos = (UMI.upper_length/2.0,-UMI.upper_height/2,0),
               height = UMI.upper_height,
               length = UMI.upper_length*1.25,
               width = 0.08,
               color = color.green)
lower_arm = box(frame = elbow_joint,
               pos = (UMI.lower_length/2.0,-UMI.lower_height/2,0),
               height = UMI.lower_height,
               length = UMI.lower_length*1.25,
               width = 0.08,
               color = color.green)

wrist = box(frame = wrist_joint,
               pos = (0,-UMI.wrist_height/8,0),
               height = UMI.wrist_height/4,
               length = 0.08,
               width = 0.08,
               color = color.green)

gripper_pos = box(frame = wrist_joint,
               pos = (0,-UMI.wrist_height/2,0.025),
               height = UMI.wrist_height,
               length = 0.03,
               width = 0.005,
               color = color.blue)

gripper_neg = box(frame = wrist_joint,
               pos = (0,-UMI.wrist_height/2,-0.025),
               height = UMI.wrist_height,
               length = 0.03,
               width = 0.005,
               color = color.blue)
gripper_open = 1

floor = box(frame=frameworld,
               pos = (0,0,0),
               height = 0.001,
               length = UMI.wpedestal + 0.6,
               width = 0.6*2,
               color = (0.5, 0.5, 0.5))
floor.pos = (floor.length/2 - UMI.wpedestal, 0, 0)
#**************************************************************************
# CHESSBOARD
# frame, board_size=0.3, position_x_z = (0.15, -0.15), angle_degrees=0)
# <<<<<<<<<<-------------------------------------------------------------------- CHANGE BOARD POSITION/ANGLE HERE
CHESSBOARD = UMI_chessboard(frameworld, 0.3, (0.15, -0.15), 0)

#***************************************************************************

# CONTROLLER Functions
def get_gripper_bottom_position():
    '''
    Gives the position of the tip of the gripper in the real world coordinate system.
    :return: Tuple in the format (x,y,z)
    '''
    return frame0.frame_to_world(
        riser.frame_to_world(
            shoulder_joint.frame_to_world(
                elbow_joint.frame_to_world(
                    wrist_joint.pos + vector(0,-UMI.wrist_height, 0)
                )
            )
        )
    )

def execute_sequence(sequence_list):
    '''
    Runs the commands as provided in a list
    :param sequence_list: List where each row contains either a GUI command or a joints-setting for the arm.
    '''
    # First move up so you do not knock over anything.
    safe_angles = deepcopy(UMI_angles)
    safe_angles[0] = CHESSBOARD.get_board_height() + 0.2 + UMI.total_arm_height
    # Set to a safe location before execution
    loop_angles = deepcopy(UMI_angles)
    # Then continue with the original plans.
    total_list = [safe_angles] + sequence_list
    chess_piece = None
    for new_angles in total_list:
        if len(new_angles) == 3 and new_angles[0] == "GUI":
            [_, command, piece_position] = new_angles
            if command == "TAKE" and chess_piece == None:
                chess_piece = CHESSBOARD.remove_piece(piece_position)
                if chess_piece != None:
                    chess_piece[0].frame=wrist_joint
                    if chess_piece[1] == "Rook":
                        piece_offset = 0
                    else:
                        piece_offset = CHESSBOARD.pieces_height[chess_piece[1]]/2
                    chess_piece[0].pos=(0,-UMI.wrist_height-piece_offset, 0)
            if command == "DROP" and chess_piece != None:
                (temp_x, temp_z) = to_coordinate(piece_position)
                if temp_x > 7 or temp_x < 0 or temp_z > 7 or temp_z < 0:
                    # Garbage field
                    chess_piece[0].visible = False
                    del chess_piece[0]
                    chess_piece = None
                else:
                    f_size = CHESSBOARD.field_size
                    chess_piece[0].frame=CHESSBOARD.framemp
                    #chess_piece[0].pos=(f_size*(temp_x+1) - f_size/2.0, 0, (f_size*temp_z) + f_size/2),
                    if chess_piece[1] == "Rook":
                        piece_offset = CHESSBOARD.pieces_height[chess_piece[1]]/2
                    else:
                        piece_offset = 0
                    chess_piece[0].pos=(f_size*(7-temp_x) + f_size/2.0, piece_offset, f_size*(7-temp_z) + f_size/2.0)
                    CHESSBOARD.pieces[piece_position] = chess_piece
                    chess_piece = None
        else:
            # Degrees to Radians.
            new_angles = [new_angles[0]] + [radians(x) for x in new_angles[1:-1]] + [new_angles[-1]]
            # Correct the height
            animate_arm(loop_angles, new_angles)
            loop_angles = deepcopy(UMI_angles)
            sleep(0.5)

def animate_arm(from_angles, to_angles):
    '''
    Given two different joint combinations, animate the movement for the arm between those two.
    :param from_angles: Original joint positions.
    :param to_angles: New joint positions.
    '''
    # Compute the differences for all joints
    old_a = np.array(from_angles)
    new_a = np.array(to_angles)
    delta_a = ( new_a - old_a )
    # Move through these differences in 100 steps
    for i in np.arange(0.0, 1.01, 0.01):
        rate(100)
        moveRiser(old_a[0] + delta_a[0]*i)
        moveShoulder(old_a[1] + delta_a[1]*i)
        moveElbow(old_a[2] + delta_a[2]*i)
        moveWrist(old_a[3] + delta_a[3]*i)
        moveGripper(old_a[4] + delta_a[4]*i)
        disp.center=get_gripper_bottom_position()

def move(chessboard, from_pos, to_pos):
    '''
    Given two positions on the board [a1-h8] compute the required actions
    :param chessboard: The chessboard object
    :param from_pos: [a1-h8]
    :param to_pos: [a1-h8]
    :return: List of actions for the simulator to run.
    '''
    sequence_list = []
    # Check if you are removing a piece from play by performing the action.
    if to_pos in chessboard.pieces:
        sequence_list += move_to_garbage(chessboard, to_pos)
    sequence_list += high_path(chessboard, from_pos, to_pos)
    # Write the output files.
    write_parameters_to_file(sequence_list, "joints_simulator.txt")
    write_parameters_to_umi_robot(sequence_list)
    return sequence_list
#**************************************************************************

# INIT CONTROLS
s0.SetValue(s0.GetMax())
s1.SetValue(0) # update the slider
s2.SetValue(0) # update the slider
s3.SetValue(0) # update the slider
s4.SetValue(50) # update the slider

# Storage only used to make the movements of the arm appear smoothed.
UMI_angles = [UMI.joint_ranges["Riser"][1], 0, 0, 0, 0.05]
#**************************************************************************


while(True):
    rate(100)
    # TIP: If you want to know at all time, what the x,y,z of your robot arm is,
    # print(get_gripper_bottom_position())
    disp.center=get_gripper_bottom_position()
#End Program
0
