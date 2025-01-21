import math
import chess
import torch
import numpy as np
from copy import deepcopy
from chess_rl import move_to_index, ChessRL

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior  # P(s,a) from the neural network
        self.children = {}  # map of move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.state = None  # Store the board tensor representation
        
    def is_leaf(self):
        return len(self.children) == 0
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
        
    def select_child(self, c_puct=1.0):
        """Select the child with the highest UCB score."""
        best_score = float('-inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # UCB score = Q + U, where U ~ P * sqrt(N) / (1 + n)
            q_value = -child.value()  # Negative because value is from opponent's perspective
            u_value = (c_puct * child.prior * 
                      math.sqrt(self.visit_count) / (1 + child.visit_count))
            ucb_score = q_value + u_value

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, board):
        """Perform MCTS search starting from the given board position."""
        root = MCTSNode(board)
        
        # Evaluate the root state with the neural network
        root.state = board_to_tensor(root.board).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            policy, value = self.model(root.state)
        policy = policy.squeeze().cpu().numpy()
        value = value.item()
        
        # Create children for all legal moves
        for move in root.board.legal_moves:
            move_idx = move_to_index(move)
            child_board = root.board.copy()
            child_board.push(move)
            root.children[move] = MCTSNode(
                child_board,
                parent=root,
                prior=policy[move_idx]
            )
        
        # Perform simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf node
            while not node.is_leaf() and not node.board.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.board.is_game_over():
                node.state = board_to_tensor(node.board).unsqueeze(0).to('cuda:0')
                with torch.no_grad():
                    policy, value = self.model(node.state)
                policy = policy.squeeze().cpu().numpy()
                value = value.item()
                
                # Create children for all legal moves
                for move in node.board.legal_moves:
                    move_idx = move_to_index(move)
                    child_board = node.board.copy()
                    child_board.push(move)
                    node.children[move] = MCTSNode(
                        child_board,
                        parent=node,
                        prior=policy[move_idx]
                    )
            else:
                # Game is over, use the game result as value
                if node.board.is_checkmate():
                    value = -1 if node.board.turn else 1
                else:
                    value = 0  # Draw
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value  # Value from opponent's perspective
        
        # Return policy vector based on visit counts
        policy = np.zeros(1968)  # Maximum possible moves
        for move, child in root.children.items():
            policy[move_to_index(move)] = child.visit_count
        
        # Temperature-adjusted policy
        policy = policy ** (1/self.temperature)
        policy = policy / np.sum(policy)
        
        return policy

class ChessRLWithMCTS(ChessRL):
    def __init__(self, num_simulations=800):
        super().__init__()
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
    
    def select_move(self, board, temperature=1.0):
        """Select a move using MCTS."""
        self.mcts.temperature = temperature
        policy = self.mcts.search(board)
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        move_probs = np.array([policy[move_to_index(move)] for move in legal_moves])
        
        # Select move based on policy
        move_idx = np.random.choice(len(legal_moves), p=move_probs/np.sum(move_probs))
        return legal_moves[move_idx]

def self_play_training_with_mcts(model, num_games=100, moves_per_game=100):
    """Train the model through self-play with MCTS."""
    for game in range(num_games):
        board = chess.Board()
        game_states = []
        mcts_policies = []
        
        for move_num in range(moves_per_game):
            if board.is_game_over():
                break
            
            # Store current state
            state = board_to_tensor(board)
            
            # Get move and policy from MCTS
            temperature = max(1.0 - move_num / 30, 0.1)
            policy = model.mcts.search(board)
            move = model.select_move(board, temperature)
            
            # Store state, policy and move
            game_states.append(state)
            mcts_policies.append(policy)
            
            # Make move
            board.push(move)
        
        # Game outcome
        if board.is_checkmate():
            value = 1.0 if board.turn else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            value = 0.0
        else:
            value = 0.0
        
        # Update replay buffer with game results
        for state, policy in zip(game_states, mcts_policies):
            model.replay_buffer.push(
                state.to('cuda:0'),
                torch.from_numpy(policy).float().to('cuda:0'),
                torch.tensor([value], device='cuda:0')
            )
            value = -value  # Flip value for opponent's moves
        
        # Training step
        if game % 5 == 0:
            loss = model.train_step(batch_size=64)
            print(f"Game {game}, Loss: {loss}")

# Usage example
if __name__ == "__main__":
    chess_rl = ChessRLWithMCTS(num_simulations=800)
    self_play_training_with_mcts(chess_rl)