import pygame
import math
import random

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

PLAYER_HEIGHT = 1.8  # Average player height (1.8 meters)
GOALKEEPER_HEIGHT = 1.9  # Goalkeeper slightly taller (1.9 meters)
GOALPOST_HEIGHT = 2.44  # Standard FIFA goalpost height (2.44 meters)

GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)  # Color for the kicking player

class Ball:
    def __init__(self, x, y, z):
        # 2D properties
        self.start_x = x
        self.start_y = y
        self.start_z = z
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.radius = 0.22 * PIXELS_PER_METER
        self.in_motion = False
        self.curve_force = 0
        
        self.gravity = 0.5  # Gravity constant (applied to z-axis)
        self.bounce_damping = 0.7  # Energy loss on bounce
        
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        self.vx = 0
        self.vy = 0
        self.vz = 0
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
        velocity_magnitude = power * 1  # sed multiplier for better range
        
        # New 3D physics implementation
        # cursor_y is negative when below center - this means hitting lower on the ball
        # Lower cursor = higher shot in real physics
        
        # Maximum height factor for highest power + lowest hit
        max_height_factor = 4.0 if power > 90 else 3.0
        
        # Base height factor calculation 
        # Lower cursor = hitting ball lower = more upward force
        # Range from 0.5 (top of ball) to max_height_factor (bottom of ball)
        height_factor = max(0.5, min(max_height_factor, (cursor_y / 20.0) + 1.0))
        
        # Power impacts overall velocity
        # XY velocity - Horizontal movement
        self.vx = base_dx * velocity_magnitude
        self.vy = base_dy * velocity_magnitude
        
        # Z velocity - Initial upward velocity based on hitting position and power
        # More power + lower hit point = higher initial velocity
        self.vz = height_factor * (power / 40)
        
        # Set curve force based on horizontal cursor position (Magnus effect)
        self.curve_force = cursor_x * 0.05
        
        self.in_motion = True
        print(f"Ball shot with power: {power}, height_factor: {height_factor:.2f}, curve: {self.curve_force:.2f}, vz: {self.vz:.2f}")
    
    def update(self):
        if self.in_motion:
            # Apply gravity to z-axis
            self.vz -= self.gravity
            
            # Apply curve force (Magnus effect)
            # The curve should be more pronounced when the ball is in the air
            # and diminish as it gets closer to the ground
            if self.z > 0:
                self.vx += self.curve_force * (self.z / 50)  # Scale curve by height
            
            # Update 3D position
            self.x += self.vx
            self.y += self.vy
            self.z += self.vz
            
            # Check if ball hits the ground
            if self.z <= 0 and self.vz < 0:
                # Bounce with energy loss
                self.z = 0
                self.vz = -self.vz * self.bounce_damping
                
                # If almost stopped, stop completely
                if abs(self.vz) < 0.5:
                    self.vz = 0
            
            # Check boundaries - return True if out of bounds
            if (self.x <= self.radius or 
                self.x >= SCREEN_WIDTH - self.radius or 
                self.y <= 0 or 
                self.y >= SCREEN_HEIGHT - self.radius):
                return True
            
            # Check if ball has stopped moving
            if (abs(self.vx) < 0.5 and 
                abs(self.vy) < 0.5 and 
                abs(self.vz) < 0.5 and
                self.z <= 0.1 and
                self.y > SCREEN_HEIGHT - 50):
                return True
                
        return False
    
    def draw(self, screen):
        # Draw shadow first (beneath the ball)
        shadow_radius = max(3, self.radius - int(self.z / 30))
        shadow_alpha = max(30, 150 - int(self.z / 5))
        
        # In a real implementation, we would create a transparent surface
        # For now, we'll simulate it by drawing a darker circle
        shadow_color = (50, 50, 50)  # Dark gray
        pygame.draw.circle(screen, shadow_color, (int(self.x), int(self.y)), shadow_radius)
        
        # Calculate visual size reduction based on height
        visual_radius = max(7, self.radius - int(self.z / 40))
        
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y - self.z/8)), visual_radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y - self.z/8)), visual_radius, 2)

class KickingPlayer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
        self.color = ORANGE
        self.animation_frame = 0
        self.is_kicking = False
        self.kick_duration = 10
        self.kick_timer = 0
        
    def start_kick(self):
        self.is_kicking = True
        self.kick_timer = 0
        self.animation_frame = 0
        
    def update(self):
        if self.is_kicking:
            self.kick_timer += 1
            self.animation_frame = min(4, self.kick_timer // 2)
            
            if self.kick_timer >= self.kick_duration:
                self.is_kicking = False
                self.kick_timer = 0
                self.animation_frame = 0
                return True
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
    # Add height property to player class
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.radius = 12
        self.color = color
        self.height = PLAYER_HEIGHT * PIXELS_PER_METER  # Height in pixels
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Goalkeeper:
    # Add height property to goalkeeper class
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 15
        self.height = 25
        self.speed = 1.5
        self.direction = 1
        self.player_height = GOALKEEPER_HEIGHT * PIXELS_PER_METER  # Height in pixels
    
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
    # This class remains the same
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
    # This class remains largely the same
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
        # Basic setup remains the same
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Football Free Kick Practice - 3D Physics")
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
        
        # Debug info
        self.show_debug = True
    
    def setup_game(self):
        # Ball position (outside penalty area)
        ball_x = SCREEN_WIDTH // 2 + random.randint(-100, 100)
        ball_y = PENALTY_AREA_HEIGHT + 80 + random.randint(0, 100)
        self.ball = Ball(ball_x, ball_y, 0)
        
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
            self.wall_players.append(Player(player_x, player_y, BLUE))
    
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
        # Use 3D goal detection
        # Check if the ball is within the goal boundaries in 2D
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        goal_top = 20
        goal_bottom = 20 + GOAL_HEIGHT
        
        # Convert ball's z-coordinate from pixels to meters
        ball_height_meters = self.ball.z / PIXELS_PER_METER
        
        if (goal_left <= self.ball.x <= goal_right and 
            goal_top <= self.ball.y <= goal_bottom):
            
            # Check if the ball is at the right height
            if ball_height_meters <= GOALPOST_HEIGHT:
                return True  # Goal!
            else:
                print(f"Ball over goalpost! Height: {ball_height_meters:.2f}m > {GOALPOST_HEIGHT}m")
        
        return False
    
    def check_collision_with_players(self):
        # Use 3D collision detection
        # Convert ball's z-coordinate from pixels to meters
        ball_height_meters = self.ball.z
        
        # Check collision with wall players
        for player in self.wall_players:
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            distance_2d = math.sqrt(dx**2 + dy**2)
            
            if distance_2d < self.ball.radius + player.radius:
                # Check if ball is high enough to pass over the player
                if ball_height_meters > PLAYER_HEIGHT:
                    if self.show_debug:
                        print(f"Ball passed over player! Height: {ball_height_meters:.2f}m > {PLAYER_HEIGHT}m")
                else:
                    print(f"Ball hit player! Height: {ball_height_meters:.2f}m <= {PLAYER_HEIGHT}m")
                    return True
        
        # Check collision with goalkeeper
        dx = self.ball.x - self.goalkeeper.x
        dy = self.ball.y - self.goalkeeper.y
        distance_2d = math.sqrt(dx**2 + dy**2)
        
        if distance_2d < self.ball.radius + self.goalkeeper.width//2:
            # Check if ball is high enough to pass over the goalkeeper
            if ball_height_meters > GOALKEEPER_HEIGHT:
                # Ball passes over the goalkeeper
                if self.show_debug:
                    print(f"Ball passed over goalkeeper! Height: {ball_height_meters:.2f}m > {GOALKEEPER_HEIGHT}m")
            else:
                # Ball hits the goalkeeper
                return True
        
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
                        # Reset ball
                        self.ball.reset()
                    
                    elif event.key == pygame.K_n and not self.ball.in_motion:
                        # New setup
                        self.setup_game()
                    
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
            
            # Update kicking player
            kick_completed = self.kicking_player.update()
            
            # If kick animation just completed and we have a shoot request, shoot the ball
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
                "N: New setup",
            ]
            
            for i, instruction in enumerate(instructions):
                text = self.font.render(instruction, True, BLACK)
                self.screen.blit(text, (10, 50 + i * 25))
            
            # Debug info
            if self.show_debug and self.ball.in_motion:
                debug_text = [
                    f"Ball height: {self.ball.z/PIXELS_PER_METER:.2f}m",
                    f"X velocity: {self.ball.vx:.2f}",
                    f"Y velocity: {self.ball.vy:.2f}",
                    f"Z velocity: {self.ball.vz:.2f}",
                    f"Curve force: {self.ball.curve_force:.2f}"
                ]
                
                for i, debug in enumerate(debug_text):
                    text = self.font.render(debug, True, BLACK)
                    self.screen.blit(text, (SCREEN_WIDTH - 200, 10 + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

game = Game()
game.run()