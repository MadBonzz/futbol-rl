import pygame
import numpy as np
import math
import random

# Initialize Pygame
pygame.init()

# Constants based on FIFA field dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 12

# Field dimensions in pixels (scaled from meters)
FIELD_WIDTH = int(68 * PIXELS_PER_METER)  # 68m field width
FIELD_HEIGHT = int(50 * PIXELS_PER_METER)  # 50m quarter field length
GOAL_WIDTH = int(9.5 * PIXELS_PER_METER)  # 9.5m goal width (enlarged)
GOAL_HEIGHT = int(3.2 * PIXELS_PER_METER)  # 3.2m goal height (enlarged)
PENALTY_AREA_WIDTH = int(40.32 * PIXELS_PER_METER)  # 40.32m penalty area width
PENALTY_AREA_HEIGHT = int(16.5 * PIXELS_PER_METER)  # 16.5m penalty area depth
WALL_MIN_DISTANCE = int(9.15 * PIXELS_PER_METER)  # 9.15m minimum wall distance

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)  # Color for the kicking player

class Ball:
    def __init__(self, x, y):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 8
        self.in_motion = False
        self.gravity = 0.5  # Increased gravity for better physics
        self.air_resistance = 0.98
        self.curve_force = 0
        
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.vx = 0
        self.vy = 0
        self.in_motion = False
        self.curve_force = 0
    
    def shoot(self, power, cursor_x, cursor_y):
        # Calculate direction towards goal center
        goal_center_x = SCREEN_WIDTH // 2
        goal_center_y = 50  # Goal is at top
        
        # Base direction towards goal
        base_dx = goal_center_x - self.x
        base_dy = goal_center_y - self.y
        
        # Normalize base direction
        base_magnitude = math.sqrt(base_dx**2 + base_dy**2)
        if base_magnitude > 0:
            base_dx /= base_magnitude
            base_dy /= base_magnitude
        
        # Convert power (0-100) to velocity magnitude
        velocity_magnitude = power * 0.4  # Increased multiplier for better range
        
        # Height adjustment based on cursor Y position (lower cursor = higher shot)
        height_factor = (-cursor_y / 30.0) + 1.0  # More height when cursor is lower
        height_factor = max(0.5, min(2.0, height_factor))  # Clamp between 0.5 and 2.0
        
        # Apply velocity with height adjustment
        self.vx = base_dx * velocity_magnitude
        self.vy = base_dy * velocity_magnitude * height_factor - 3  # Subtract for upward initial velocity
        
        # Set curve force based on horizontal cursor position
        self.curve_force = cursor_x * 0.05  # Curve strength
        
        self.in_motion = True
        print(f"Ball shot with power: {power}, height_factor: {height_factor:.2f}, curve: {self.curve_force:.2f}")
    
    def update(self):
        if self.in_motion:
            # Apply gravity
            self.vy += self.gravity
            
            # Apply curve force (Magnus effect)
            self.vx += self.curve_force
            
            # Update position
            self.x += self.vx
            self.y += self.vy
            
            # Apply air resistance
            self.vx *= self.air_resistance
            self.vy *= self.air_resistance
            
            # Check boundaries - return True if out of bounds
            if (self.x <= self.radius or self.x >= SCREEN_WIDTH - self.radius or 
                self.y <= 0 or self.y >= SCREEN_HEIGHT - self.radius):
                return True
            
            # Check if ball has stopped moving
            if abs(self.vx) < 0.5 and abs(self.vy) < 0.5 and self.y > SCREEN_HEIGHT - 50:
                return True
                
        return False
    
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 2)

class KickingPlayer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
        self.color = ORANGE  # Different color from wall players
        self.animation_frame = 0
        self.is_kicking = False
        self.kick_duration = 10  # frames
        self.kick_timer = 0
        
    def start_kick(self):
        self.is_kicking = True
        self.kick_timer = 0
        self.animation_frame = 0
        
    def update(self):
        if self.is_kicking:
            self.kick_timer += 1
            self.animation_frame = min(4, self.kick_timer // 2)  # 5 frames over 10 ticks
            
            if self.kick_timer >= self.kick_duration:
                self.is_kicking = False
                self.kick_timer = 0
                self.animation_frame = 0
                return True  # Kick completed
        return False
    
    def draw(self, screen):
        # Draw body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw head
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y - 20)), 8)
        
        # Draw kicking leg based on animation frame
        if self.is_kicking:
            leg_offset = self.animation_frame * 3
            leg_x = self.x + leg_offset
            leg_y = self.y + 10
            pygame.draw.line(screen, self.color, (self.x, self.y + 5), (leg_x, leg_y), 5)
        else:
            # Normal stance
            pygame.draw.line(screen, self.color, (self.x - 5, self.y + 5), (self.x - 5, self.y + 20), 5)
            pygame.draw.line(screen, self.color, (self.x + 5, self.y + 5), (self.x + 5, self.y + 20), 5)

class Player:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.radius = 12
        self.color = color
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Goalkeeper:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 15
        self.height = 25
        self.speed = 1.5  # Slower movement as requested
        self.direction = 1
    
    def update(self):
        # Simple AI movement
        self.x += self.speed * self.direction
        
        # Bounce within goal area
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = (SCREEN_WIDTH + GOAL_WIDTH) // 2
        
        if self.x <= goal_left + self.width//2 or self.x >= goal_right - self.width//2:
            self.direction *= -1
    
    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, 
                        (self.x - self.width//2, self.y - self.height//2, 
                         self.width, self.height))

class PowerBar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 200
        self.height = 20
        self.power = 50  # 0-100
        self.max_power = 100
    
    def increase_power(self):
        self.power = min(self.max_power, self.power + 5)
    
    def decrease_power(self):
        self.power = max(0, self.power - 5)
    
    def draw(self, screen, font):
        # Draw bar background
        pygame.draw.rect(screen, GRAY, (self.x, self.y, self.width, self.height))
        
        # Draw power level
        power_width = (self.power / self.max_power) * self.width
        color = GREEN if self.power < 70 else YELLOW if self.power < 90 else RED
        pygame.draw.rect(screen, color, (self.x, self.y, power_width, self.height))
        
        # Draw border
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        
        # Draw text
        text = font.render(f"Power: {self.power}%", True, BLACK)
        screen.blit(text, (self.x, self.y - 25))

class DirectionControl:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ball_radius = 30
        self.cursor_x = 0
        self.cursor_y = -10  # Start slightly up
        self.cursor_radius = 5
    
    def move_cursor(self, dx, dy):
        new_x = self.cursor_x + dx
        new_y = self.cursor_y + dy
        
        # Keep cursor within ball
        distance = math.sqrt(new_x**2 + new_y**2)
        if distance <= self.ball_radius - self.cursor_radius:
            self.cursor_x = new_x
            self.cursor_y = new_y
    
    def get_direction(self):
        return self.cursor_x, self.cursor_y
    
    def draw(self, screen, font):
        # Draw ball representation
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.ball_radius)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.ball_radius, 2)
        
        # Draw center cross
        pygame.draw.line(screen, GRAY, (self.x - 5, self.y), (self.x + 5, self.y), 1)
        pygame.draw.line(screen, GRAY, (self.x, self.y - 5), (self.x, self.y + 5), 1)
        
        # Draw cursor
        cursor_screen_x = self.x + self.cursor_x
        cursor_screen_y = self.y + self.cursor_y
        pygame.draw.circle(screen, RED, (int(cursor_screen_x), int(cursor_screen_y)), self.cursor_radius)
        
        # Draw text
        text = font.render("Direction Control", True, BLACK)
        screen.blit(text, (self.x - 60, self.y - 60))
        
        # Draw instruction text
        text1 = font.render("Lower = Higher Shot", True, BLACK)
        text2 = font.render("Left/Right = Curve", True, BLACK)
        screen.blit(text1, (self.x - 80, self.y + 40))
        screen.blit(text2, (self.x - 80, self.y + 55))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Football Free Kick Practice")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        
        # Game objects
        self.setup_game()
        
        # UI elements
        self.power_bar = PowerBar(50, SCREEN_HEIGHT - 100)
        self.direction_control = DirectionControl(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        
        # Game state
        self.score = 0
        self.attempts = 0
        self.game_over = False
        self.shoot_requested = False
    
    def setup_game(self):
        # Ball position (outside penalty area)
        ball_x = SCREEN_WIDTH // 2 + random.randint(-100, 100)
        ball_y = PENALTY_AREA_HEIGHT + 80 + random.randint(0, 100)
        self.ball = Ball(ball_x, ball_y)
        
        # Kicking player positioned behind the ball
        self.kicking_player = KickingPlayer(ball_x, ball_y + 25)
        
        # Goalkeeper
        goal_center_x = SCREEN_WIDTH // 2
        self.goalkeeper = Goalkeeper(goal_center_x, 50)
        
        # Wall players (2-6 players, at least 9.15m from ball)
        wall_size = random.randint(2, 6)
        wall_distance = max(WALL_MIN_DISTANCE, random.randint(WALL_MIN_DISTANCE, WALL_MIN_DISTANCE + 60))
        wall_y = self.ball.y - wall_distance
        
        self.wall_players = []
        wall_width = wall_size * 30
        start_x = self.ball.x - wall_width // 2
        
        for i in range(wall_size):
            player_x = start_x + i * 30 + random.randint(-5, 5)
            player_y = wall_y + random.randint(-10, 10)
            self.wall_players.append(Player(player_x, player_y, BLUE))  # Blue color for wall players
    
    def draw_field(self):
        # Field background
        self.screen.fill(GREEN)
        
        # Goal area
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        
        # Goal posts
        pygame.draw.rect(self.screen, WHITE, (goal_left - 5, 20, 10, GOAL_HEIGHT))
        pygame.draw.rect(self.screen, WHITE, (goal_right - 5, 20, 10, GOAL_HEIGHT))
        
        # Goal line
        pygame.draw.line(self.screen, WHITE, (goal_left, 20 + GOAL_HEIGHT), 
                        (goal_right, 20 + GOAL_HEIGHT), 3)
        
        # Penalty area
        penalty_left = (SCREEN_WIDTH - PENALTY_AREA_WIDTH) // 2
        penalty_right = penalty_left + PENALTY_AREA_WIDTH
        pygame.draw.rect(self.screen, WHITE, 
                        (penalty_left, 20, PENALTY_AREA_WIDTH, PENALTY_AREA_HEIGHT), 3)
        
        # Center line (partial)
        center_y = SCREEN_HEIGHT // 2
        pygame.draw.line(self.screen, WHITE, (0, center_y), (SCREEN_WIDTH, center_y), 3)
    
    def check_goal(self):
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        goal_top = 20
        goal_bottom = 20 + GOAL_HEIGHT
        
        if (goal_left <= self.ball.x <= goal_right and 
            goal_top <= self.ball.y <= goal_bottom):
            return True
        return False
    
    def check_collision_with_players(self):
        for player in self.wall_players:
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.ball.radius + player.radius:
                return True  # Hit wall player
        
        # Check collision with goalkeeper
        dx = self.ball.x - self.goalkeeper.x
        dy = self.ball.y - self.goalkeeper.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.ball.radius + self.goalkeeper.width//2:
            return True  # Hit goalkeeper
        
        return False
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.ball.in_motion and not self.kicking_player.is_kicking:
                        # Start the kick animation
                        self.kicking_player.start_kick()
                        self.shoot_requested = True
                        print("Space pressed - starting kick animation")
                    
                    elif event.key == pygame.K_r and not self.ball.in_motion:
                        self.ball.reset()
                    
                    elif event.key == pygame.K_n and not self.ball.in_motion:
                        self.setup_game()
                    
                    elif event.key == pygame.K_UP:
                        self.power_bar.increase_power()
                    elif event.key == pygame.K_DOWN:
                        self.power_bar.decrease_power()
                    
                    elif event.key == pygame.K_w:
                        self.direction_control.move_cursor(0, -3)
                    elif event.key == pygame.K_s:
                        self.direction_control.move_cursor(0, 3)
                    elif event.key == pygame.K_a:
                        self.direction_control.move_cursor(-3, 0)
                    elif event.key == pygame.K_d:
                        self.direction_control.move_cursor(3, 0)
            
            # Update kicking player
            kick_completed = self.kicking_player.update()
            
            if kick_completed and self.shoot_requested:
                dx, dy = self.direction_control.get_direction()
                self.ball.shoot(self.power_bar.power, dx, dy)
                self.attempts += 1
                self.shoot_requested = False
                print("Ball shot executed!")
            
            # Update game objects
            ball_out_of_bounds = self.ball.update()
            self.goalkeeper.update()
            
            # Check for various end conditions
            should_reset = False
            
            if self.ball.in_motion:
                # Check for goal
                if self.check_goal():
                    self.score += 1
                    should_reset = True
                    print("GOAL!")
                
                # Check for collisions
                elif self.check_collision_with_players():
                    should_reset = True
                    print("Hit player!")
                
                # Check if ball went out of bounds
                elif ball_out_of_bounds:
                    should_reset = True
                    print("Out of bounds!")
            
            # Auto-reset if needed
            if should_reset:
                pygame.time.wait(1000)  # Brief pause to see the result
                self.setup_game()
            
            # Draw everything
            self.draw_field()
            
            # Draw game objects
            for player in self.wall_players:
                player.draw(self.screen)
            
            self.goalkeeper.draw(self.screen)
            self.kicking_player.draw(self.screen)
            self.ball.draw(self.screen)
            
            # Draw UI
            self.power_bar.draw(self.screen, self.font)
            self.direction_control.draw(self.screen, self.font)
            
            # Draw score and instructions
            score_text = self.big_font.render(f"Goals: {self.score}/{self.attempts}", True, BLACK)
            self.screen.blit(score_text, (10, 10))
            
            # Instructions
            instructions = [
                "SPACE: Shoot",
                "↑↓: Power",
                "WASD: Direction",
                "R: Reset ball",
                "N: New setup"
            ]
            
            for i, instruction in enumerate(instructions):
                text = self.font.render(instruction, True, BLACK)
                self.screen.blit(text, (10, 50 + i * 25))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

# Create and run the game
if __name__ == "__main__":
    game = Game()
    game.run()