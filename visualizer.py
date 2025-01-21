import chess
import chess.svg
import cairosvg
import os
from datetime import datetime
import json
from IPython.display import clear_output, display
import time

__all__ = ['visualize_game']

class ChessVisualizer:
    def __init__(self, output_dir="game_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_pgn(self, board, game_number):
        """Save game in PGN format"""
        game_path = os.path.join(self.output_dir, f"game_{game_number}.pgn")
        with open(game_path, "w") as f:
            f.write(str(board.game()))
    
    def save_frames(self, board, game_number):
        """Save each position as an SVG/PNG frame"""
        game_dir = os.path.join(self.output_dir, f"game_{game_number}_frames")
        os.makedirs(game_dir, exist_ok=True)
        
        # First save as SVG
        svg_str = chess.svg.board(board=board)
        svg_path = os.path.join(game_dir, f"move_{len(board.move_stack):03d}.svg")
        with open(svg_path, "w") as f:
            f.write(svg_str)
        
        # Convert to PNG if needed
        png_path = svg_path.replace(".svg", ".png")
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=png_path)
        
    def create_game_video(self, game_number):
        """Create MP4 video from PNG frames using ffmpeg"""
        game_dir = os.path.join(self.output_dir, f"game_{game_number}_frames")
        video_path = os.path.join(self.output_dir, f"game_{game_number}.mp4")
        
        os.system(f'ffmpeg -framerate 2 -i {game_dir}/move_%03d.png '
                 f'-c:v libx264 -pix_fmt yuv420p {video_path}')
    
    def print_live_board(self, board):
        """Print the current board state in console"""
        clear_output(wait=True)
        print(board)
        time.sleep(0.5)  # Add delay to make it readable
    
    def save_game_json(self, board, game_number, metadata=None):
        """Save game in JSON format with additional metadata"""
        game_data = {
            "moves": [str(move) for move in board.move_stack],
            "result": board.result(),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        json_path = os.path.join(self.output_dir, f"game_{game_number}.json")
        with open(json_path, "w") as f:
            json.dump(game_data, f, indent=2)

def visualize_game(board, game_number, methods=None):
    """Wrapper function to visualize a game using multiple methods"""
    if methods is None:
        methods = ["pgn"]  # Default to PGN only
        
    viz = ChessVisualizer()
    
    if "pgn" in methods:
        viz.save_pgn(board, game_number)
    
    if "frames" in methods or "video" in methods:
        viz.save_frames(board, game_number)
        
    if "video" in methods:
        viz.create_game_video(game_number)
    
    if "json" in methods:
        viz.save_game_json(board, game_number)