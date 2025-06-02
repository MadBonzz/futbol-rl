# DQN Training Interface for Football Free Kick Game
import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

class DQNTrainer:
    def __init__(self, env, model, optimizer, loss_fn, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the DQN trainer.
        
        Args:
            env: DQN environment wrapper
            model: DQN model (you need to provide this)
            optimizer: PyTorch optimizer
            loss_fn: Loss function (usually MSE)
            device: Device to run on
        """
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Training parameters
        self.target_update_freq = 100  # Update target network every N episodes
        self.train_freq = 4  # Train every N steps
        self.step_count = 0
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.goals_scored = []
        self.walls_passed = []
        
        print(f"DQN Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action for exploration
            return random.randrange(self.env.action_space)
        else:
            # Greedy action from model
            with torch.no_grad():
                state_tensor = state.unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        Perform one training step using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self):
        """
        Train for one complete episode.
        
        Returns:
            episode_info: Dictionary with episode statistics
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        goal_scored = False
        wall_passed = False
        
        while True:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            self.step_count += 1
            
            # Train if enough steps
            if self.step_count % self.train_freq == 0:
                self.train_step()
            
            # Move to next state
            state = next_state
            
            if done:
                # Check final outcome
                if info.get('shot_taken', False):
                    # Analyze shot result
                    wall_passed, goal_scored, _ = self.env.game.collision_detector.comprehensive_analysis(
                        self.env.game.ball, self.env.game.wall_players, self.env.game.goalkeeper
                    )
                break
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.goals_scored.append(1 if goal_scored else 0)
        self.walls_passed.append(1 if wall_passed else 0)
        
        episode_info = {
            'reward': episode_reward,
            'length': episode_length,
            'goal_scored': goal_scored,
            'wall_passed': wall_passed,
            'epsilon': self.epsilon
        }
        
        return episode_info
    
    def train(self, num_episodes, save_freq=100, render_freq=50):
        """
        Train the DQN for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            save_freq: Frequency to save model
            render_freq: Frequency to render episodes
        """
        print(f"Starting training for {num_episodes} episodes...")
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Train episode
            episode_info = self.train_episode()
            
            # Render occasionally
            if episode % render_freq == 0:
                self.env.render()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_goals = np.mean(self.goals_scored[-10:]) if self.goals_scored else 0
                avg_walls = np.mean(self.walls_passed[-10:]) if self.walls_passed else 0
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_info['reward']:6.2f} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Goal%: {avg_goals:.2f} | "
                      f"Wall%: {avg_walls:.2f} | "
                      f"ε: {self.epsilon:.3f}")
            
            # Save model periodically
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f"dqn_model_episode_{episode}.pth")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_model("dqn_model_final.pth")
        
        # Plot training results
        self.plot_training_results()
    
    def evaluate(self, num_episodes=10, render=True):
        """
        Evaluate the trained model.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            
        Returns:
            evaluation_results: Dictionary with evaluation statistics
        """
        print(f"Evaluating model for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_goals = []
        eval_walls = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            goal_scored = False
            wall_passed = False
            
            while True:
                # Select action (no exploration)
                action = self.select_action(state, training=False)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
                    time.sleep(0.05)  # Slow down for visibility
                
                state = next_state
                
                if done:
                    if info.get('shot_taken', False):
                        wall_passed, goal_scored, _ = self.env.game.collision_detector.comprehensive_analysis(
                            self.env.game.ball, self.env.game.wall_players, self.env.game.goalkeeper
                        )
                    break
            
            eval_rewards.append(episode_reward)
            eval_goals.append(1 if goal_scored else 0)
            eval_walls.append(1 if wall_passed else 0)
            
            print(f"Eval Episode {episode+1}: Reward={episode_reward:.2f}, "
                  f"Goal={'Yes' if goal_scored else 'No'}, "
                  f"Wall Passed={'Yes' if wall_passed else 'No'}")
        
        results = {
            'avg_reward': np.mean(eval_rewards),
            'goal_rate': np.mean(eval_goals),
            'wall_pass_rate': np.mean(eval_walls),
            'rewards': eval_rewards
        }
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Goal Rate: {results['goal_rate']:.2%}")
        print(f"Wall Pass Rate: {results['wall_pass_rate']:.2%}")
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'goals_scored': self.goals_scored,
            'walls_passed': self.walls_passed
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.goals_scored = checkpoint.get('goals_scored', [])
        self.walls_passed = checkpoint.get('walls_passed', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_results(self):
        """
        Plot training results.
        """
        if not self.episode_rewards:
            print("No training data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # Moving average of rewards
        window = min(100, len(self.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, color='red', alpha=0.7)
        
        # Episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        # Goal rate
        if len(self.goals_scored) > 10:
            goal_rate = np.convolve(self.goals_scored, np.ones(10)/10, mode='valid')
            ax3.plot(range(9, len(self.goals_scored)), goal_rate)
        ax3.set_title('Goal Rate (10-episode moving average)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Goal Rate')
        ax3.set_ylim(0, 1)
        
        # Wall pass rate
        if len(self.walls_passed) > 10:
            wall_rate = np.convolve(self.walls_passed, np.ones(10)/10, mode='valid')
            ax4.plot(range(9, len(self.walls_passed)), wall_rate)
        ax4.set_title('Wall Pass Rate (10-episode moving average)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Wall Pass Rate')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.show()


class DQNGameRunner:
    """
    Modified game runner that uses DQN for gameplay instead of human input.
    """
    
    def __init__(self, game, trainer, render=True):
        """
        Initialize the DQN game runner.
        
        Args:
            game: Game instance
            trainer: Trained DQN trainer
            render: Whether to render the game
        """
        self.game = game
        self.trainer = trainer
        self.render_enabled = render
        self.clock = pygame.time.Clock()
        
    def run_autonomous(self, num_episodes=10):
        """
        Run the game autonomously using the trained DQN.
        
        Args:
            num_episodes: Number of episodes to run
        """
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            self.run_single_episode()
    
    def run_single_episode(self):
        """
        Run a single episode using DQN.
        """
        # Reset environment
        state = self.trainer.env.reset()
        
        episode_reward = 0
        step_count = 0
        
        while True:
            # Get action from DQN (no exploration)
            action = self.trainer.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, info = self.trainer.env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Render if enabled
            if self.render_enabled:
                self.trainer.env.render()
                self.clock.tick(10)  # Control speed
            
            # Print action taken
            action_meanings = self.trainer.env.get_action_meanings()
            print(f"Step {step_count}: {action_meanings[action]} (Reward: {reward:.3f})")
            
            state = next_state
            
            if done:
                # Print episode summary
                print(f"Episode finished!")
                print(f"Total steps: {step_count}")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Shot taken: {info.get('shot_taken', False)}")
                
                if info.get('shot_taken', False):
                    # Get shot result
                    wall_passed, goal_scored, analysis = self.trainer.env.game.collision_detector.comprehensive_analysis(
                        self.trainer.env.game.ball, 
                        self.trainer.env.game.wall_players, 
                        self.trainer.env.game.goalkeeper
                    )
                    print(f"Wall passed: {wall_passed}")
                    print(f"Goal scored: {goal_scored}")
                    print(f"Analysis: {analysis}")
                
                break
    
    def run_interactive_demo(self):
        """
        Run an interactive demo where user can trigger DQN episodes.
        """
        print("Interactive DQN Demo")
        print("Press SPACE to run DQN episode, ESC to quit")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        print("\nRunning DQN episode...")
                        self.run_single_episode()
                        print("Press SPACE for another episode, ESC to quit")
            
            # Render current state
            if self.render_enabled:
                self.trainer.env.render()
                self.clock.tick(60)
        
        pygame.quit()