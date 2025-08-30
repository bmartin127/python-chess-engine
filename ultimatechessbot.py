import chess
import itertools
import chess.svg
from IPython.display import SVG, display
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime
import numpy as np

# All possible ranks and files on a chessboard
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

# All possible source and destination squares
squares = [file + rank for file, rank in itertools.product(files, ranks)]

# Possible promotion pieces (pawn promotions can only promote to these)
promotions = ['q', 'r', 'b', 'n']


# Function to generate all possible UCI moves
def generate_uci_moves():
    moves = []
    # Generate all regular piece moves
    for src in squares:
        for dest in squares:
            if src != dest:
                moves.append(f"{src}{dest}")

    # Handle pawn promotions
    for file in files:
        for rank in ['7']:  # Pawns promote from the 7th rank
            src = f"{file}{rank}"
            dest_rank = '8'  # Promote to the 8th rank
            for piece in promotions:
                dest = f"{file}{dest_rank}"
                moves.append(f"{src}{dest}{piece}")

    return moves


def board_to_tensor(board):
    # Create a 8x8 tensor initialized to zero
    tensor = torch.zeros((8, 8), dtype=torch.float32)

    # Mapping of pieces to values
    piece_map = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }

    # Fill the tensor based on the board's pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_map[piece.piece_type]
            # Use negative value for black pieces
            if piece.color == chess.BLACK:
                value = -value
            tensor[square // 8][square % 8] = value

    return tensor


# Define a simple feedforward neural network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # Input layer
        self.fc2 = nn.Linear(128, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, 128)  # Hidden layer
        self.fc4 = nn.Linear(128, 64)  # Hidden layer
        self.fc5 = nn.Linear(64, 4064)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Generate the moves
uci_moves = generate_uci_moves()
model = ChessNet()


def generatemove(board):
    chess_tensor = board_to_tensor(board)
    input_tensor = chess_tensor.view(-1)  # Flattened tensor
    output = model(input_tensor)
    top_move = get_top_move(output, uci_moves, board)
    return top_move


def get_top_move(output, uci_moves, board):
    # Create a mask for legal moves
    legal_moves = [move.uci() for move in board.legal_moves]
    mask = torch.zeros_like(output)

    # Set the mask to 1 for legal moves
    for move in legal_moves:
        if move in uci_moves:
            index = uci_moves.index(move)
            mask[index] = 1

    # Apply the mask to the output
    output = output * mask

    # Ensure no illegal moves have non-zero probabilities
    output = torch.where(mask == 1, output, torch.tensor(float('-inf'), dtype=output.dtype))

    # Get the index of the move with the highest output value
    top_move_index = torch.argmax(output).item()

    return uci_moves[top_move_index]


# Initialize model and optimizer
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def save_board_as_png(board: chess.Board):
    # Generate a filename based on the board's FEN
    fen = board.fen()
    # Replace special characters in FEN to create a valid filename
    safe_fen = re.sub(r'[\/ ]', '_', fen)  # Replace '/' and ' ' with '_'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp
    svg_filename = f"chess_board_{safe_fen}_{timestamp}.svg"  # Generate filename

    # Render the board to SVG
    svg_data = chess.svg.board(board)

    # Save the SVG to a file
    with open(svg_filename, "w") as f:
        f.write(svg_data)

    print(f"Board saved as {svg_filename}")


# Self-play loop
def self_play():
    print("Self playing")
    board = chess.Board()
    rewards = []
    states = []
    move_count = 0  # To keep track of the number of moves made

    while not board.is_game_over():
        # Get the current board state as tensor
        chess_tensor = board_to_tensor(board).view(-1)

        # Forward pass through the network to get move probabilities
        top_move = generatemove(board)

        # Attempt to make the move on the board and check if it's legal
        try:
            board.push_uci(top_move)
            # Store the state
            states.append(chess_tensor)
            move_count += 1
        except ValueError:
            # If the move is illegal, penalize with a negative reward and skip state storage
            print(f"Illegal move attempted: {top_move}")
            rewards.append(-1.0)  # Penalize for illegal move
            continue  # Skip to the next iteration

        # Calculate reward based on the game state at the end of the game
        if board.is_checkmate():
            print("Checkmate!")
            reward = 1.0 if board.turn == chess.WHITE else -1.0  # Based on the losing side
            save_board_as_png(board)
        elif board.is_stalemate():
            print("Stalemate!")
            reward = 0.0  # Draw
        elif board.is_insufficient_material():
            print("Insufficient material!")
            reward = 0.0  # Draw
        elif board.can_claim_fifty_moves():
            print("Fifty-move rule!")
            reward = 0.0  # Draw
        elif board.can_claim_threefold_repetition():
            print("Threefold repetition!")
            reward = -0.5  # Slight penalty for threefold repetition
        else:
            reward = 0.0  # Neutral reward for ongoing game

        states.append(chess_tensor)
        rewards.append(reward)

    # Reset board after game ends
    board.reset()
    return states, rewards


def train():
    print("Beginning training")
    for epoch in range(10000):  # Number of epochs or games played
        print("epoch: ", str(epoch))
        states, rewards = self_play()

        # Normalize rewards
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)  # Normalize rewards

        for state, reward in zip(states, rewards):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(state)

            # Create a tensor of the same shape as the model's output, filled with the scalar reward
            target = torch.full_like(output, reward, dtype=torch.float32)

            # Compute loss (difference between predicted output and actual reward)
            loss = criterion(output, target)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), "chess_model.pth")  # Save model to a file
    print("Model saved to chess_model.pth")


train()
