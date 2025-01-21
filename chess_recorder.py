import chess
import chess.svg
import cairosvg
import os
import shutil

__all__ = ['save_game_video', 'ChessVideoRecorder']

class ChessVideoRecorder:
    def __init__(self, output_dir="game_videos"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_frames_dir = None
        self.current_game = None
    
    def start_game(self, game_number):
        """Start recording a new game"""
        self.current_game = game_number
        self.current_frames_dir = os.path.join(self.output_dir, f"temp_frames_{game_number}")
        os.makedirs(self.current_frames_dir, exist_ok=True)
    
    def save_frame(self, board):
        """Save the current board position as a frame"""
        if self.current_frames_dir is None:
            return
            
        position_number = len(board.move_stack)
        frame_path = os.path.join(self.current_frames_dir, f"position_{position_number:04d}.png")
        svg_str = chess.svg.board(board=board)
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=frame_path)
    
    def end_game(self, framerate=1):
        """Create video from collected frames and cleanup"""
        if self.current_frames_dir is None:
            return
            
        # Create video from frames
        frames_pattern = os.path.join(self.current_frames_dir, "position_%04d.png")
        video_path = os.path.join(self.output_dir, f"game_{self.current_game}.mp4")
        
        os.system(f'ffmpeg -y -framerate {framerate} -i {frames_pattern} '
                 f'-c:v libx264 -pix_fmt yuv420p {video_path} -hide_banner -loglevel error')
        
        # Cleanup
        shutil.rmtree(self.current_frames_dir)
        self.current_frames_dir = None
        self.current_game = None

def save_game_video(board, game_number, framerate=1):
    """Wrapper function to easily save a game as video"""
    video_maker = ChessVideoRecorder()
    video_maker.start_game(game_number)
    video_maker.save_frame(board)
    video_maker.end_game(framerate)