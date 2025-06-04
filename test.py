import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from freekick import Game
from trainer import HansiFlick
import torch

def train(trainer : HansiFlick, game : Game, n_episodes = 100, train_intervals=5):
    if not pygame.get_init():
        pygame.init()

    agent = trainer

    run_history = {
        'state' : [],
        'action' : [],
        'reward' : [],
        'next_state' : []
    }

    screen = pygame.display.set_mode((800, 600))

    for i in range(n_episodes):
        if (i+1) % train_intervals == 0:
            agent.train_model(run_history)
            run_history = {
                            'state' : [],
                            'action' : [],
                            'reward' : [],
                            'next_state' : []
                        }
        game.screen.fill((34, 139, 34))  # Fill with GREEN
        game.draw_field()
        game.draw_game_objects()
        game.draw_ui()
        pygame.display.flip()
        current_episode = True

        frame = game.get_frame_as_tensor()
        processed_frame = agent.process_frame(frame)
        while current_episode:
            game.screen.fill((34, 139, 34))  # Fill with GREEN
            game.draw_field()
            game.draw_game_objects()
            game.draw_ui()
            pygame.display.flip()
            run_history['state'].append(processed_frame)
            action = agent.predict(processed_frame)
            game.step(action)
            reward = 0
            ball_finished = False
            if action == 6:
                while not ball_finished:
                    ball_finished = game.ball.update()
                    game.kicking_player.update()
                    game.goalkeeper.update()
                    game.screen.fill((34, 139, 34))  # Fill with GREEN
                    game.draw_field()
                    game.draw_game_objects()
                    game.draw_ui()
                    pygame.display.flip()
                    pygame.time.delay(100)
                    if ball_finished and game.ball.trajectory_points:
                        final_dist, _, goal = game.handle_shot_result()
                        if goal:
                            reward = 1
                        else:
                            reward = 1 / final_dist
                        print(reward)
                        run_history['action'].append(action)
                        run_history['reward'].append(reward)
                        run_history['next_state'].append(None)
                        current_episode = False
                        break
            else:
                ball_finished = game.ball.update()
                game.kicking_player.update()
                game.goalkeeper.update()
                frame = game.get_frame_as_tensor()
                processed_frame = agent.process_frame(frame)
                run_history['action'].append(action)
                run_history['reward'].append(reward)
                run_history['next_state'].append(processed_frame)
                pygame.time.delay(100)
        game.setup_scenario()
        agent.epsilon *= agent.decay_factory
    pygame.quit()
    torch.save(trainer.model, 'final.pth')

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
        reward = -0.01
        ball_finished = game.ball.update()
        game.kicking_player.update()
        game.goalkeeper.update()
        
        # If the ball has finished its trajectory, set up a new scenario
        if ball_finished and game.ball.trajectory_points:
            wall_passed, goal = game.handle_shot_result()
            game.setup_scenario()
            
        # Add a small delay to make visualization possible
        pygame.time.delay(100)
    
    pygame.quit()
    return all_frames, all_actions

if __name__ == "__main__":
    # print("\nRunning game with AI agent...")
    # frames, actions = run_game_with_ai(num_frames=200, display_every=5)
    
    # print(f"Processed {len(frames)} frames with corresponding actions")
    # print("Game session complete")
    game = Game()
    trainer = HansiFlick()
    train(trainer, game, 10000)