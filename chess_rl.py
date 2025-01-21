import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
from chess_recorder import ChessVideoRecorder

def get_move_space_size():
    """Return the total size of the move space."""
    regular_moves = 64 * 64
    promotion_moves_per_piece = 64 * 8
    promotion_pieces = 4  # Queen, Rook, Bishop, Knight
    promotion_moves = promotion_moves_per_piece * promotion_pieces
    return regular_moves + promotion_moves

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.policy_head = nn.Linear(512, get_move_space_size())
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (torch.stack(states).to('cuda:0'), 
                torch.stack(policies).to('cuda:0'),
                torch.stack(values).to('cuda:0'))
    
    def __len__(self):
        return len(self.buffer)

def board_to_tensor(board):
    """Convert chess board to tensor representation"""
    pieces = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
             'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    tensor = torch.zeros(8, 8, 8)
    
    # Set piece planes
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = i // 8, i % 8
            piece_idx = pieces[piece.symbol()]
            color_idx = 0 if piece.color else 1
            plane_idx = piece_idx // 2
            tensor[plane_idx][rank][file] = 1 if color_idx == 0 else -1
            
    # Add repetition planes
    tensor[6] = torch.ones(8, 8) if board.turn else torch.zeros(8, 8)
    tensor[7] = torch.ones(8, 8) * len(board.move_stack) / 100.0
    
    return tensor.permute(0, 1, 2)

class ChessRL:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer()

    def move_to_index(self, move):
        """Convert a chess move to an index in the policy vector.
        Regular moves: first 4096 indices (64*64)
        Promotion moves: next 2048 indices (4 pieces * 64 * 8)
        """
        src = move.from_square
        dst = move.to_square
        promotion = move.promotion
        
        # Base index for the move
        index = src * 64 + dst
        
        # Handle promotions
        if promotion:
            promotion_base = 64 * 64
            if promotion == chess.QUEEN:
                index = promotion_base + (src * 8 + dst % 8)
            elif promotion == chess.ROOK:
                index = promotion_base + (64 * 8) + (src * 8 + dst % 8)
            elif promotion == chess.BISHOP:
                index = promotion_base + (2 * 64 * 8) + (src * 8 + dst % 8)
            elif promotion == chess.KNIGHT:
                index = promotion_base + (3 * 64 * 8) + (src * 8 + dst % 8)
                
        # Safety check
        if index >= get_move_space_size():
            print(f"Warning: move {move} produced invalid index {index}")
            index = 0
            
        return index

    def select_move(self, board, temperature=1.0):
        """Select a move using the current policy"""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)
        
        legal_moves = list(board.legal_moves)
        move_probs = torch.zeros(len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            move_idx = self.move_to_index(move)
            if move_idx < policy[0].size(0):
                move_probs[i] = policy[0][move_idx]
        
        # Apply temperature and handle zero probabilities
        move_probs = move_probs ** (1/temperature)
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
        else:
            move_probs = torch.ones_like(move_probs) / len(move_probs)
        
        move_idx = torch.multinomial(move_probs, 1).item()
        return legal_moves[move_idx]
    
    def train_step(self, batch_size=32):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, policies, values = self.replay_buffer.sample(batch_size)
        
        # Forward pass
        pred_policies, pred_values = self.model(states)
        
        # Ensure policies are the same size
        if pred_policies.size(1) > policies.size(1):
            padding = torch.zeros(policies.size(0), 
                                pred_policies.size(1) - policies.size(1),
                                device=self.device)
            policies = torch.cat([policies, padding], dim=1)
        
        # Calculate losses
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / batch_size
        value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

def self_play_training(model, num_games=1000000, moves_per_game=1000, viz_every=50):
    """Train the model through self-play with video recording"""
    recorder = ChessVideoRecorder()
    
    for game in range(num_games):
        board = chess.Board()
        game_states = []
        should_record = game % viz_every == 0
        
        if should_record:
            recorder.start_game(game)
            recorder.save_frame(board)
        
        for move_num in range(moves_per_game):
            if board.is_game_over():
                break
            
            # Store current state
            state = board_to_tensor(board)
            
            # Select and make move
            temperature = max(1.0 - move_num / 30, 0.1)
            move = model.select_move(board, temperature)
            board.push(move)
            
            # Store state and move
            game_states.append((state, move))
            
            # Record position if we're recording this game
            if should_record:
                recorder.save_frame(board)
        
        # Finish recording if we were recording this game
        if should_record:
            recorder.end_game(framerate=2)
            print(f"\nVideo saved for game {game}!")
        
        # Game outcome
        if board.is_checkmate():
            value = 1.0 if board.turn else -1.0
            outcome = "Checkmate"
        elif board.is_stalemate():
            value = 0.0
            outcome = "Stalemate"
        elif board.is_insufficient_material():
            value = 0.0
            outcome = "Insufficient material"
        else:
            value = 0.0
            outcome = "Move limit"
        
        if should_record:
            print(f"Game {game} ended by: {outcome}")
        
        # Update replay buffer
        for state, move in game_states:
            model.replay_buffer.push(state, 
                                  torch.zeros(get_move_space_size()).index_fill_(0, 
                                      torch.tensor(model.move_to_index(move)), 1.0),
                                  torch.tensor([value]))
            value = -value  # Flip value for opponent's moves
        
        # Training step
        if game % 10 == 0:
            loss = model.train_step()
            print(f"Game {game}, Loss: {loss}")

if __name__ == "__main__":
    chess_rl = ChessRL()
    self_play_training(chess_rl)