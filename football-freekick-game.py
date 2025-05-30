import pygame
import math
import random
import sys

# Initialize Pygame
pygame.init()

# Constants based on FIFA field dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 12

# Field dimensions in pixels (scaled from meters)
FIELD_WIDTH = int(68 * PIXELS_PER_METER)  # 68m field width
FIELD_HEIGHT = int(50 * PIXELS_PER_METER)  # 50m quarter field length
GOAL_WIDTH = int(9.5 * PIXELS_PER_METER)  # Increased from 7.32m to 9.5m
GOAL_HEIGHT = int(3.2 * PIXELS_PER_METER)  # Increased from 2.44m to 3.2m
PENALTY_AREA_WIDTH = int(40.32 * PIXELS_PER_METER)  # 40.32m penalty area width
PENALTY_AREA_HEIGHT = int(16.5 * PIXELS_PER_METER)  # 16.5m penalty area depth

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

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
        self.gravity = 0.5
        self.air_resistance = 0.98
        self.curve_factor = 0  # For side spin (Magnus effect)
        
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.vx = 0
        self.vy = 0
        self.in_motion = False
        self.curve_factor = 0
    
    def shoot(self, power, direction_y, direction_x=0):
        """
        Shoot the ball with given power and direction
        power: 0-100 scale for shot power
        direction_y: Vertical aim (negative = higher shot)
        direction_x: Horizontal aim for curve (0 = straight, negative = left curve, positive = right curve)
        """
        # Base velocity scaled by power (0-100)
        base_velocity = power * 0.25
        
        # Calculate trajectory based on aiming position
        # Lower position on ball (more negative direction_y) = higher trajectory
        angle = math.atan2(-direction_y, 15)  # Fixed forward direction with variable height
        
        # Set velocity components
        self.vx = math.cos(angle) * base_velocity
        self.vy = math.sin(angle) * base_velocity
        
        # Set curve based on horizontal aim
        self.curve_factor = direction_x * 0.015
        
        self.in_motion = True
    
    def update(self):
        if self.in_motion:
            # Apply gravity
            self.vy += self.gravity
            
            # Apply curve (Magnus effect)
            self.vx += self.curve_factor
            
            # Update position
            self.x += self.vx
            self.y += self.vy
            
            # Apply air resistance
            self.vx *= self.air_resistance
            self.vy *= self.air_resistance
            
            # Check if ball stopped
            if abs(self.vx) < 0.1 and abs(self.vy) < 0.1 and self.y >= SCREEN_HEIGHT - self.radius - 5:
                self.in_motion = False
                return "stopped"
            
            # Check boundaries
            if self.x <= self.radius or self.x >= SCREEN_WIDTH - self.radius:
                return "out"
            
            if self.y >= SCREEN_HEIGHT - self.radius:
                self.vy *= -0.5  # Bounce with energy loss
                self.y = SCREEN_HEIGHT - self.radius
                
                # Slow down horizontal movement on bounce
                self.vx *= 0.8
                
            return "in_play"
    
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 2)

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 50
        self.kicking = False
        self.kick_frame = 0
        self.kick_frames = 5  # Total frames in kicking animation
        
        # Load player images (simple rectangle representation for now)
        # In a real game, you would load actual sprite images here
        self.standing_img = self.create_player_surface(BLUE)
        
        # Create simple kicking animation frames
        self.kick_images = []
        for i in range(self.kick_frames):
            # Create different poses for kicking animation
            angle = i * 20  # Degrees of leg movement
            self.kick_images.append(self.create_kicking_surface(BLUE, angle))
        
        self.current_img = self.standing_img
    
    def create_player_surface(self, color):
        # Create a simple player representation (rectangle with a circle head)
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Body
        pygame.draw.rect(surface, color, (5, 10, 20, 40))
        # Head
        pygame.draw.circle(surface, color, (15, 10), 10)
        return surface
    
    def create_kicking_surface(self, color, leg_angle):
        # Create a simple kicking animation frame
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Body
        pygame.draw.rect(surface, color, (5, 5, 20, 30))
        # Head
        pygame.draw.circle(surface, color, (15, 5), 10)
        
        # Convert angle to radians
        rad_angle = math.radians(leg_angle)
        
        # Draw kicking leg
        leg_length = 20
        end_x = 15 + math.cos(rad_angle) * leg_length
        end_y = 35 + math.sin(rad_angle) * leg_length
        pygame.draw.line(surface, color, (15, 35), (end_x, end_y), 8)
        
        return surface
    
    def start_kick(self):
        self.kicking = True
        self.kick_frame = 0
    
    def update(self):
        if self.kicking:
            # Update kick animation
            self.kick_frame += 1
            if self.kick_frame >= self.kick_frames:
                self.kicking = False
                self.current_img = self.standing_img
            else:
                self.current_img = self.kick_images[self.kick_frame]
        else:
            self.current_img = self.standing_img
    
    def draw(self, screen):
        screen.blit(self.current_img, (int(self.x - self.width/2), int(self.y - self.height/2)))

class Goalkeeper:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 15
        self.height = 25
        self.speed = 1.5  # Reduced from 3.0 to 1.5
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
        
        # Add center marker
        self.show_center = True
    
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
        
        # Draw center marker
        if self.show_center:
            pygame.draw.line(screen, GRAY, (self.x - 5, self.y), (self.x + 5, self.y), 1)
            pygame.draw.line(screen, GRAY, (self.x, self.y - 5), (self.x, self.y + 5), 1)
        
        # Draw cursor
        cursor_screen_x = self.x + self.cursor_x
        cursor_screen_y = self.y + self.cursor_y
        pygame.draw.circle(screen, RED, (int(cursor_screen_x), int(cursor_screen_y)), self.cursor_radius)
        
        # Draw direction arrow
        if self.cursor_x != 0 or self.cursor_y != 0:
            end_x = self.x - self.cursor_x * 2
            end_y = self.y - self.cursor_y * 2
            pygame.draw.line(screen, RED, (self.x, self.y), (end_x, end_y), 3)
        
        # Draw text
        text = font.render("Direction Control", True, BLACK)
        screen.blit(text, (self.x - 60, self.y - 60))
        
        # Draw height/curve explanation
        height_text = font.render("Lower = Higher Shot", True, BLACK)
        curve_text = font.render("Left/Right = Curve", True, BLACK)
        screen.blit(height_text, (self.x - 60, self.y + 40))
        screen.blit(curve_text, (self.x - 60, self.y + 60))

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
    
    def setup_game(self):
        # Ball position (outside penalty area)
        ball_x = SCREEN_WIDTH // 2
        ball_y = PENALTY_AREA_HEIGHT + 50 + random.randint(0, 100)
        self.ball = Ball(ball_x, ball_y)
        
        # Player position (behind the ball)
        self.player = Player(ball_x - 30, ball_y)
        
        # Goalkeeper
        goal_center_x = SCREEN_WIDTH // 2
        self.goalkeeper = Goalkeeper(goal_center_x, 50)
        
        # Wall players (2-6 players, at least 10m from ball)
        wall_size = random.randint(2, 6)
        wall_distance = max(120, random.randint(120, 150))  # At least 10m (120 pixels)
        wall_y = self.ball.y - wall_distance
        
        self.wall_players = []
        wall_width = wall_size * 30
        start_x = self.ball.x - wall_width // 2
        
        for i in range(wall_size):
            player_x = start_x + i * 30 + random.randint(-5, 5)
            player_y = wall_y + random.randint(-10, 10)
            self.wall_players.append(Player(player_x, player_y))
    
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
        # Check collision with wall players
        for player in self.wall_players:
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.ball.radius + player.width/2:
                return True
        
        # Check collision with goalkeeper
        dx = self.ball.x - self.goalkeeper.x
        dy = self.ball.y - self.goalkeeper.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.ball.radius + self.goalkeeper.width/2:
            return True
            
        return False
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.ball.in_motion:
                        # Shoot the ball
                        dx, dy = self.direction_control.get_direction()
                        self.player.start_kick()
                        
                        # Delay the actual shot by a few frames to sync with kick animation
                        pygame.time.set_timer(pygame.USEREVENT, 200)
                        
                    # Power controls
                    elif event.key == pygame.K_UP:
                        self.power_bar.increase_power()
                    elif event.key == pygame.K_DOWN:
                        self.power_bar.decrease_power()
                    
                    # Direction controls
                    elif event.key == pygame.K_w:
                        self.direction_control.move_cursor(0, -3)
                    elif event.key == pygame.K_s:
                        self.direction_control.move_cursor(0, 3)
                    elif event.key == pygame.K_a:
                        self.direction_control.move_cursor(-3, 0)
                    elif event.key == pygame.K_d:
                        self.direction_control.move_cursor(3, 0)
            
            # Check for the kick timer event
            if pygame.event.get(pygame.USEREVENT):
                if self.player.kicking and not self.ball.in_motion:
                    # Actually shoot the ball when the kick animation reaches the right frame
                    dx, dy = self.direction_control.get_direction()
                    self.ball.shoot(self.power_bar.power, dy, dx)
                    self.attempts += 1
                    pygame.time.set_timer(pygame.USEREVENT, 0)  # Cancel the timer
            
            # Update game objects
            self.player.update()
            self.ball.update()
            self.goalkeeper.update()
            
            # Check collisions and goals
            if self.ball.in_motion:
                # Check for collisions with wall or goalkeeper
                if self.check_collision_with_players():
                    # Automatic reset on collision
                    self.setup_game()
                
                # Check for goal
                if self.check_goal():
                    self.score += 1
                    # Automatic reset on goal
                    self.setup_game()
                
                # Check if ball is out of bounds or stopped
                ball_status = self.ball.update()
                if ball_status in ["out", "stopped"]:
                    # Automatic reset
                    self.setup_game()
            
            # Draw everything
            self.draw_field()
            
            # Draw game objects
            for player in self.wall_players:
                player.draw(self.screen)
            
            self.goalkeeper.draw(self.screen)
            self.ball.draw(self.screen)
            self.player.draw(self.screen)
            
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
            ]
            
            for i, instruction in enumerate(instructions):
                text = self.font.render(instruction, True, BLACK)
                self.screen.blit(text, (10, 50 + i * 25))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# Create and run the game
if __name__ == "__main__":
    game = Game()
    game.run()