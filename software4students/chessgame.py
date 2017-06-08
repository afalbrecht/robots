from __future__ import print_function
from copy import deepcopy
import math
import sys

## Helper functions

# Translate a position in chess notation to x,y-coordinates
# Example: c3 corresponds to (2,5)
def to_coordinate(notation):
    x = ord(notation[0]) - ord('a')
    y = 8 - int(notation[1])
    return (x, y)

# Translate a position in x,y-coordinates to chess notation
# Example: (2,5) corresponds to c3
def to_notation(coordinates):
    (x,y) = coordinates
    letter = chr(ord('a') + x)
    number = 8 - y
    return letter + str(number)

# Translates two x,y-coordinates into a chess move notation
# Example: (1,4) and (2,3) will become b4c5
def to_move(from_coord, to_coord):
    return to_notation(from_coord) + to_notation(to_coord)

## Defining board states

# These Static classes are used as enums for:
# - Material.Rook
# - Material.King
# - Material.Pawn
# - Side.White
# - Side.Black
class Material:
    Rook, King, Pawn, Bishop, Queen, Knight = ['r','k','p','b','q','t']
class Side:
    White, Black = range(0,2)

# A chesspiece on the board is specified by the side it belongs to and the type
# of the chesspiece
class Piece:
    def __init__(self, side, material):
        self.side = side
        self.material = material


# A chess configuration is specified by whose turn it is and a 2d array
# with all the pieces on the board
class ChessBoard:
    
    def __init__(self, turn):
        # This variable is either equal to Side.White or Side.Black
        self.turn = turn
        self.board_matrix = None


    ## Getter and setter methods 
    def set_board_matrix(self,board_matrix):
        self.board_matrix = board_matrix

    # Note: assumes the position is valid
    def get_boardpiece(self,position):
        (x,y) = position
        return self.board_matrix[y][x]

    # Note: assumes the position is valid
    def set_boardpiece(self,position,piece):
        (x,y) = position
        self.board_matrix[y][x] = piece
    
    # Read in the board_matrix using an input string
    def load_from_input(self,input_str):
        self.board_matrix = [[None for _ in range(8)] for _ in range(8)]
        x = 0
        y = 0
        for char in input_str:
            if y == 8:
                if char == 'W':
                    self.turn = Side.White
                elif char == 'B':
                    self.turn = Side.Black
                return
            if char == '\r':
                continue
            if char == '.':
                x += 1
                continue
            if char == '\n':
                x = 0
                y += 1
                continue 
            
            if char.isupper():
                side = Side.White
            else:
                side = Side.Black
            material = char.lower()

            piece = Piece(side, material)
            self.set_boardpiece((x,y),piece)
            x += 1

    # Print the current board state
    def __str__(self):
        return_str = ""
        y = 8
        for board_row in self.board_matrix:
            return_str += str(y) + "  " 
            for piece in board_row:
                if piece == None:
                    return_str += ". "
                else:
                    char = piece.material
                    if piece.side == Side.White:
                        char = char.upper()
                    return_str += char + ' '
            return_str += '\n'
            y -= 1

        return_str += "   A B C D E F G H\n"
        turn_name = ("White" if self.turn == Side.White else "Black") 
        return_str += "It is " + turn_name + "'s turn\n"

        return return_str

    # Given a move string in chess notation, return a new ChessBoard object
    # with the new board situation
    # Note: this method assumes the move suggested is a valid, legal move
    def make_move(self, move_str):
        
        start_pos = to_coordinate(move_str[0:2])
        end_pos = to_coordinate(move_str[2:4])

        if self.turn == Side.White:
            turn = Side.Black
        else:
            turn = Side.White
            
        # Duplicate the current board_matrix
        new_matrix = [row[:] for row in self.board_matrix]
        
        # Create a new chessboard object
        new_board = ChessBoard(turn)
        new_board.set_board_matrix(new_matrix)

        # Carry out the move in the new chessboard object
        piece = new_board.get_boardpiece(start_pos)
        new_board.set_boardpiece(end_pos, piece)
        new_board.set_boardpiece(start_pos, None)

        return new_board

    # Added clause for checkmate
    def is_king_dead(self, side):
        seen_king = False
        for x in range(8):
            for y in range(8):
                piece = self.get_boardpiece((x,y))
                if piece != None and piece.side == side and piece.material == Material.King:
                    seen_king = True
                    if  to_notation((x,y)) in self.legal_movescheck(side, kingbool=True) and \
                        self.move_king(x,y) == []:
                        seen_king = False
        return not seen_king

    # Part of pawn implementation
    def piece_in_front(self, x, y):
        if self.turn == Side.White:
            check_y = y - 1
            if self.get_boardpiece((x, check_y)) is not None:
                return True
            return False
        else:
            check_y = y + 1
            if self.get_boardpiece((x, check_y)) is not None:
                return True
            return False

    # Part of pawn implementation
    def enemy_piece_left_front(self, x, y):
        if self.turn == Side.White:
            check_x = x - 1
            check_y = y - 1
            if self.get_boardpiece((check_x, check_y)) is None:
                return False
            if self.get_boardpiece((check_x, check_y)).side is self.turn:
                return False
            else:
                return True
        else:
            check_x = x - 1
            check_y = y + 1
            if self.get_boardpiece((check_x, check_y)) is None:
                return False
            if self.get_boardpiece((check_x, check_y)).side is self.turn:
                return False
            else:
                return True

    # Part of pawn implementation
    def enemy_piece_right_front(self, x, y):
        if self.turn == Side.White:
            check_x = x + 1
            check_y = y - 1
            if self.get_boardpiece((check_x, check_y)) is None:
                return False
            if self.get_boardpiece((check_x, check_y)).side is self.turn:
                return False
            else:
                return True
        else:
            check_x = x + 1
            check_y = y + 1
            if self.get_boardpiece((check_x, check_y)) is None:
                return False
            if self.get_boardpiece((check_x, check_y)).side is self.turn:
                return False
            else:
                return True

    # Part of pawn implementation
    def out_of_bounds_pawn(self, x, y):
        if self.turn == Side.White:
            if y == 0:
                return True
        if self.turn == Side.Black:
            if y == 8:
                return True
        else:
            return False

    def move_pawn(self, x, y):
        legal_moves = []
        if self.turn == Side.White:
            possible_moves = [(x, y - 1), (x - 1, y - 1), (x + 1, y - 1)]
            if y == 0:
                return legal_moves
            if not ChessBoard.piece_in_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[0]))
            if ChessBoard.enemy_piece_left_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[1]))
            if ChessBoard.enemy_piece_right_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[2]))

        else:
            possible_moves = [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]
            if y == 7:
                return legal_moves
            if not ChessBoard.piece_in_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[0]))
            if ChessBoard.enemy_piece_left_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[1]))
            if ChessBoard.enemy_piece_right_front(self, x, y):
                legal_moves.append(to_move((x,y),possible_moves[2]))
        return legal_moves

    # Movecheck function to check possible moves for the other side
    def legal_movescheck(self, side, kingbool=False):
        movelist = []
        output = []
        for x in range(8):
            for y in range(8):
                piece = self.get_boardpiece((x,y))
                if piece == None or piece.side is side:
                    continue
                elif piece.material == Material.Pawn:
                    movelist += self.move_pawn(x,y)
                elif piece.material == Material.King:
                    movelist += self.move_king(x,y, kingcheck=kingbool)
                elif piece.material == Material.Rook:
                    movelist += self.move_rook(x,y)
                elif piece.material == Material.Bishop:
                    movelist += self.move_bishop(x,y)
                elif piece.material == Material.Queen:
                    movelist += self.move_queen(x,y)
                elif piece.material == Material.Knight:
                    movelist += self.move_knight(x,y)
        for move in movelist:
            output.append(move[2:4])
        return output

    # Checks if surrounding moves do not end in a check
    # Kincheck variable to stop recursion
    def move_king(self,x,y, kingcheck=True):
        moves = []
        for x_c in range(-1, 2):
            change_x = x + x_c
            if change_x < 0 or change_x > 7:
                continue
            for y_c in range(-1, 2):
                change_y = y + y_c
                if change_y < 0 or change_y > 7:
                    continue
                if self.get_boardpiece((change_x,change_y)) is not None and \
                        self.get_boardpiece((change_x, change_y)).side is self.turn:
                    continue
                if kingcheck and to_notation((change_x,change_y)) in self.legal_movescheck(self.turn, kingbool=False):
                    continue
                moves.append(to_move((x,y),(change_x,change_y)))
        return moves

    def move_rook(self,x,y):
        side = self.get_boardpiece((x,y)).side
        moves = []
        x_moves = [i for i in range(8)]
        delx = []
        for x_c in x_moves:
            if self.get_boardpiece((x_c, y)) is not None:
                if x_c < x:
                    delx += x_moves[:x_c]
                if x_c > x:
                    delx += x_moves[x_c + 1:]
                if self.get_boardpiece((x_c, y)).side == side:
                    delx.append(x_c)
        for x_i in x_moves:
            if x_i not in delx:
                moves.append(to_move((x, y), (x_i, y)))
        y_moves = [i for i in range(8)]
        dely = []
        for y_c in y_moves:
            if self.get_boardpiece((x, y_c)) is not None:
                if y_c < y:
                    dely += y_moves[:y_c]
                if y_c > y:
                    dely += y_moves[y_c+1:]
                if self.get_boardpiece((x, y_c)).side == side:
                    dely.append(y_c)
        for y_i in y_moves:
            if y_i not in dely:
                moves.append(to_move((x, y), (x, y_i)))
        return moves

    def move_bishop(self,x,y):
        side = self.get_boardpiece((x, y)).side
        moves = []
        x1 = x + 1
        y1 = y + 1
        while x1 <= 7 and y1 <= 7:
            if self.get_boardpiece((x1, y1)) is None:
                moves.append(to_move((x, y), (x1, y1)))
                x1 += 1
                y1 += 1
            elif self.get_boardpiece((x1, y1)).side != side:
                moves.append(to_move((x, y), (x1, y1)))
                break
            else:
                break
        x1 = x + 1
        y1 = y - 1
        while x1 <= 7 and y1 >= 0:
            if self.get_boardpiece((x1, y1)) is None:
                moves.append(to_move((x, y), (x1, y1)))
                x1 += 1
                y1 -= 1
            elif self.get_boardpiece((x1, y1)).side != side:
                moves.append(to_move((x, y), (x1, y1)))
                break
            else:
                break
        x1 = x - 1
        y1 = y + 1
        while x1 >= 0 and y1 <= 7:
            if self.get_boardpiece((x1, y1)) is None:
                moves.append(to_move((x, y), (x1, y1)))
                x1 -= 1
                y1 += 1
            elif self.get_boardpiece((x1, y1)).side != side:
                moves.append(to_move((x, y), (x1, y1)))
                break
            else:
                break
        x1 = x - 1
        y1 = y - 1
        while x1 >= 0 and y1 >= 0:
            if self.get_boardpiece((x1, y1)) is None:
                moves.append(to_move((x, y), (x1, y1)))
                x1 -= 1
                y1 -= 1
            elif self.get_boardpiece((x1, y1)).side != side:
                moves.append(to_move((x, y), (x1, y1)))
                break
            else:
                break
        return moves

    def move_queen(self, x, y):
        output = []
        output += self.move_bishop(x, y)
        output += self.move_rook(x,y)
        return output

    def move_knight(self, x, y):
        side = self.get_boardpiece((x, y)).side
        intermediate_list = []
        moves = []
        cor_list = [-2, -1, 1, 2]
        ill_list = [-4, -2, 0, 2, 4]
        for xval in cor_list:
            for yval in cor_list:
                if (xval + yval) not in ill_list:
                    newx = x + xval
                    newy = y + yval
                    newpos = (newx, newy)
                    if 0 <= newx <= 7 and 0 <= newy <= 7:
                        if self.get_boardpiece(newpos) is None:
                            intermediate_list.append(newpos)
                        else:
                            if self.get_boardpiece(newpos).side is not side:
                                intermediate_list.append(newpos)
        for coord in intermediate_list:
            moves.append(to_move((x, y), coord))
        return moves

    # This function should return, given the current board configuration and
    # which players turn it is, all the moves possible for that player
    # It should return these moves as a list of move strings, e.g.
    # [c2c3, d4e5, f4f8]
    # TODO: write an implementation for this function
    def legal_moves(self):
        movelist = []
        for x in range(8):
            for y in range(8):
                piece = self.get_boardpiece((x,y))
                if piece == None or piece.side is not self.turn:
                    continue
                elif piece.material == Material.Pawn:
                    movelist += self.move_pawn(x,y)
                elif piece.material == Material.King:
                    movelist += self.move_king(x,y)
                elif piece.material == Material.Rook:
                    movelist += self.move_rook(x,y)
                elif piece.material == Material.Bishop:
                    movelist += self.move_bishop(x,y)
                elif piece.material == Material.Queen:
                    movelist += self.move_queen(x,y)
                elif piece.material == Material.Knight:
                    movelist += self.move_knight(x,y)
        return movelist

    # This function should return, given the move specified (in the format
    # 'd2d3') whether this move is legal
    # TODO: write an implementation for this function, implement it in terms
    # of legal_moves()
    def is_legal_move(self, move):
        if move in self.legal_moves():
            return True
        return False


# This static class is responsible for providing functions that can calculate
# the optimal move using minimax
class ChessComputer:

    # Calculates the score of a given board configuration based on the
    # material left on the board. Returns a score number, in which positive
    # means white is better off, while negative means black is better off
    # Adds a 1100 penalty when a side can't make any moves
    @staticmethod
    def evaluate_board(chessboard, depth_left):
        score = 0
        score_dict = {Material.Pawn:10, Material.Knight:30, Material.Bishop:30,
                      Material.Rook:50, Material.Queen:90, Material.King:1000}
        for x in range(8):
            for y in range(8):
                piece = chessboard.get_boardpiece((x,y))
                if piece is not None:
                    if piece.side is Side.White:
                        score += score_dict[piece.material]
                    else:
                        score -= score_dict[piece.material]
        if chessboard.turn is Side.White:
            score = score + depth_left
            if chessboard.legal_movescheck(Side.Black, kingbool=True) is []:
                score += 1100
        else:
            score = score - depth_left
            if chessboard.legal_movescheck(Side.White, kingbool=True) is []:
                score -= 1100
        return score

    # This method uses either alphabeta or minimax to calculate the best move
    # possible. The input needed is a chessboard configuration and the max
    # depth of the search algorithm. It returns a tuple of (score, chessboard)
    # with score the maximum score attainable and chessboardmove that is needed
    #to achieve this score.
    @staticmethod
    def computer_move(chessboard, depth, alphabeta=False):
        if alphabeta:
            inf = 99999999
            min_inf = -inf
            return ChessComputer.alphabeta(chessboard, depth, min_inf, inf)
        else:
            return ChessComputer.minimax(chessboard, depth)


    # This function uses minimax to calculate the next move. Given the current
    # chessboard and max depth, this function should return a tuple of the
    # the score and the move that should be executed
    # NOTE: use ChessComputer.evaluate_board() to calculate the score
    # of a specific board configuration after the max depth is reached
    # TODO: write an implementation for this function
    @staticmethod
    def min_value(chessboard, depth):
        move_list = ChessBoard.legal_moves(chessboard)
        if depth == 0 or chessboard.is_king_dead(0) or chessboard.is_king_dead(1) or move_list == []:
            return ChessComputer.evaluate_board(chessboard, depth)
        best_value = 9999999
        for move in move_list:
            new_board = chessboard.make_move(move)
            value = ChessComputer.max_value(new_board, depth - 1)
            best_value = min(value, best_value)
        return best_value

    @staticmethod
    def max_value(chessboard, depth):
        move_list = ChessBoard.legal_moves(chessboard)
        if depth == 0 or chessboard.is_king_dead(0) or chessboard.is_king_dead(1) or move_list == []:
            return ChessComputer.evaluate_board(chessboard, depth)
        best_value = -99999999
        for move in move_list:
            new_board = chessboard.make_move(move)
            value = ChessComputer.min_value(new_board, depth - 1)
            best_value = max(best_value, value)
        return best_value

    @staticmethod
    def minimax(chessboard, depth):
        movelist = ChessBoard.legal_moves(chessboard)
        if chessboard.turn is Side.White:  #max
            best_value = -99999999
            best_move = None
            for move in movelist:
                new_board = chessboard.make_move(move)
                value = ChessComputer.min_value(new_board, depth - 1)
                print(value)
                if value > best_value:
                    best_value = value
                    best_move = move
            return (best_value, best_move)

        else:  #min
            best_value = 999999999
            best_move = None
            for move in movelist:
                new_board = chessboard.make_move(move)
                value = ChessComputer.max_value(new_board, depth - 1)
                if  value < best_value:
                    best_value = value
                    best_move = move
            return (best_value, best_move)

    # This function uses alphabeta to calculate the next move. Given the
    # chessboard and max depth, this function should return a tuple of the
    # the score and the move that should be executed.
    # It has alpha and beta as extra pruning parameters
    # NOTE: use ChessComputer.evaluate_board() to calculate the score
    # of a specific board configuration after the max depth is reached
    @staticmethod
    def alphabeta(chessboard, depth, alpha, beta):
        movelist = ChessBoard.legal_moves(chessboard)
        if chessboard.turn is Side.White:  # max
            best_value = -99999999
            best_move = None
            for move in movelist:
                new_board = chessboard.make_move(move)
                value = ChessComputer.min_value_ab(new_board, alpha, beta, depth - 1)
                if value > best_value:
                    best_value = value
                    best_move = move
            return (best_value, best_move)

        else:  # min
            best_value = 999999999
            best_move = None
            for move in movelist:
                new_board = chessboard.make_move(move)
                value = ChessComputer.max_value_ab(new_board, alpha, beta, depth - 1)
                if value < best_value:
                    best_value = value
                    best_move = move
            return (best_value, best_move)

    @staticmethod
    def max_value_ab(chessboard, alpha, beta, depth):
        move_list = ChessBoard.legal_moves(chessboard)
        if depth == 0 or chessboard.is_king_dead(0) or chessboard.is_king_dead(1) or move_list == []:
            return ChessComputer.evaluate_board(chessboard, depth)
        value = -99999999
        for move in moves:
            new_board = chessboard.make_move(move)
            value = max(value, ChessComputer.min_value_ab(new_board, alpha, beta, depth - 1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    @staticmethod
    def min_value_ab(chessboard, alpha, beta, depth):
        move_list = ChessBoard.legal_moves(chessboard)
        if depth == 0 or chessboard.is_king_dead(0) or chessboard.is_king_dead(1) or move_list == []:
            return ChessComputer.evaluate_board(chessboard, depth)
        value = 99999999
        for move in moves:
            new_board = chessboard.make_move(move)
            value = min(value, ChessComputer.max_value_ab(new_board, alpha, beta, depth - 1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value


# This class is responsible for starting the chess game, playing and user 
# feedback
class ChessGame:
    def __init__(self, turn):
     
        # NOTE: you can make this depth higher once you have implemented
        # alpha-beta, which is more efficient
        self.depth = 4
        self.chessboard = ChessBoard(turn)

        # If a file was specified as commandline argument, use that filename
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        else:
            filename = "board.chb"

        print("Reading from " + filename + "...")
        self.load_from_file(filename)

    def load_from_file(self, filename):
        with open(filename) as f:
            content = f.read()

        self.chessboard.load_from_input(content)

    def main(self):
        while True:
            print(self.chessboard)

            # Print the current score
            score = ChessComputer.evaluate_board(self.chessboard,self.depth)
            print("Current score: " + str(score))

            # Calculate the best possible move
            new_score, best_move = self.make_computer_move()
            if best_move is None:
                print("Stalemate reached")
            else:
                print("Best move: " + best_move)
                print("Score to achieve: " + str(new_score))
            print("")
            self.make_human_move()


    def make_computer_move(self):
        print("Calculating best move...")
        return ChessComputer.computer_move(self.chessboard,
                self.depth, alphabeta=False)
        

    def make_human_move(self):
        # Endlessly request input until the right input is specified
        while True:
            if sys.version_info[:2] <= (2, 7):
                move = raw_input("Indicate your move (or q to stop): ")
            else:
                move = input("Indicate your move (or q to stop): ")
            if move == "q":
                print("Exiting program...")
                sys.exit(0)
            elif self.chessboard.is_legal_move(move):
                break
            print("Incorrect move!")

        self.chessboard = self.chessboard.make_move(move)

        # Exit the game if one of the kings is dead
        if self.chessboard.is_king_dead(Side.Black):
            print(self.chessboard)
            print("White wins!")
            sys.exit(0)
        elif self.chessboard.is_king_dead(Side.White):
            print(self.chessboard)
            print("Black wins!")
            sys.exit(0)

chess_game = ChessGame(Side.White)
chess_game.main()

