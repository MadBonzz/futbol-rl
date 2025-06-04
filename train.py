import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from freekick import Game
from pg_trainer import HansiFlick
import torch

def train(trainer : HansiFlick, game : Game, n_episodes = 100, train_intervals=5):
    if not pygame.get_init():
        pygame.init()

    agent = trainer

    screen = pygame.display.set_mode((800, 600))

    for i in range(n_episodes):
        run_history = {
            'state' : [],
            'action' : [],
            'reward' : [],
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
            probs = agent.predict(processed_frame)
            action = torch.multinomial(probs, 1).item()
            run_history['state'].append(processed_frame)
            run_history['action'].append(action)
            game.step(action)
            reward = -.001
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
                        run_history['reward'].append(reward)
                        current_episode = False
                        trainer.train_model(run_history)
                        break
            else:
                ball_finished = game.ball.update()
                game.kicking_player.update()
                game.goalkeeper.update()
                frame = game.get_frame_as_tensor()
                processed_frame = agent.process_frame(frame)
                run_history['reward'].append(reward)
                pygame.time.delay(100)
        game.setup_scenario()
    pygame.quit()
    torch.save(trainer.model, 'pg_final.pth')


game = Game()
trainer = HansiFlick()
train(trainer, game, 10000)