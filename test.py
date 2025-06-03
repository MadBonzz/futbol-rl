import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from freekick import Game
from model import LamineYamal
import torch
from torch.optim import Adam
from torch import nn

class HansiFlick:
    def __init__(self, input_shape=(3, 600, 800), num_actions=7, epsilon = 1, epsilon_factor=0.995, lr=1e-3, device='cuda'):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.model = LamineYamal(input_shape[0], input_shape[1], input_shape[2], num_actions)
        self.model = self.model.to(device)
        self.lr = lr
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.decay_factory = epsilon_factor
        
    def predict(self, frame_tensor):
        prob = np.random.random()
        self.epsilon *= self.decay_factory
        if prob <= self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            frame_tensor = frame_tensor.to(self.device)
            out = self.model(frame_tensor)
            return torch.argmax(out).item()
        
        
    def process_frame(self, frame_tensor):
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0
        return frame_tensor

def display_tensor(tensor, title="Game Frame"):
    img = tensor.squeeze(0)  
    if tensor.is_cuda or tensor.requires_grad:
        img = img.detach().cpu()
    
    img = img.permute(1, 2, 0)  # Shape becomes (H, W, 3)
    
    img_np = img.numpy()
    
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Ensure values are in valid range [0, 1]
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # Create image from uint8 array
    image = Image.fromarray(img_uint8)
    
    image.save('frame.png')
    plt.figure(figsize=(12, 9))
    plt.imshow(img_np)
    plt.axis('off')  # Hide axes
    plt.title(title)
    plt.show()

def run_game_with_ai(num_frames=100, save_frames=False, display_every=10):
    if not pygame.get_init():
        pygame.init()
        
    # Create game instance
    game = Game()
    
    # Create deep learning agent
    agent = HansiFlick()
    
    # Store frames if needed
    all_frames = []
    all_actions = []
    
    # Create screen to render game
    screen = pygame.display.set_mode((800, 600))
    
    # Process frames
    for i in range(num_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        game.screen.fill((34, 139, 34))  # Fill with GREEN
        game.draw_field()
        game.draw_game_objects()
        game.draw_ui()
        pygame.display.flip()
        
        # Get the frame as tensor
        frame = game.get_frame_as_tensor()
        
        # Store frame if needed
        if save_frames:
            all_frames.append(frame)
            
        # Display frame occasionally
        # if i % display_every == 0:
        #     display_tensor(frame, f"Frame {i}")
            
        # Process frame with deep learning model
        processed_frame = agent.process_frame(frame)
        action = agent.predict(processed_frame)
        all_actions.append(action)
        
        print(f"Frame {i}: Predicted action {action}")
        
        # Update game state based on model's prediction
        game.step(action)
        
        # Update ball and other game objects
        ball_finished = game.ball.update()
        game.kicking_player.update()
        game.goalkeeper.update()
        
        # If the ball has finished its trajectory, set up a new scenario
        if ball_finished and game.ball.trajectory_points:
            game.handle_shot_result()
            game.setup_scenario()
            
        # Add a small delay to make visualization possible
        pygame.time.delay(100)
    
    pygame.quit()
    return all_frames, all_actions

if __name__ == "__main__":
    # Run the game with AI for 20 frames
    print("\nRunning game with AI agent...")
    frames, actions = run_game_with_ai(num_frames=200, display_every=5)
    
    print(f"Processed {len(frames)} frames with corresponding actions")
    print("Game session complete")