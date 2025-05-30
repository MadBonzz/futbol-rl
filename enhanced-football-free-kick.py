import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 12
GRAVITY = 0.5
AIR_RESISTANCE = 0.98
BALL_MASS = 0.43  # Standard football mass in kg

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.vx = 0
        self.vy = 0
        self.radius = 8
        self.in_motion = False
        self.spin = 0  # Angular velocity in rad/s
        self.lift_coefficient = 0

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.vx = 0
        self.vy = 0
        self.in_motion = False
        self.spin = 0
        self.lift_coefficient = 0

    def shoot(self, power, cursor_x, cursor_y):
        # Calculate relative cursor position (-1 to 1 for both axes)
        rel_x = (cursor_x - self.x) / self.radius
        rel_y = (cursor_y - self.y) / self.radius
        
        # Calculate spin based on vertical position
        vertical_factor = rel_y  # -1 (bottom) to 1 (top)
        self.spin = 15 * vertical_factor  # Backspin when hitting bottom
        
        # Calculate initial velocity components
        velocity_magnitude = power * 0.3
        base_angle = math.atan2(SCREEN_HEIGHT//2 - self.y, SCREEN_WIDTH//2 - self.x)
        
        # Apply horizontal deviation for curve
        horizontal_factor = rel_x
        curve_angle = horizontal_factor * math.pi/4  # Max 45 degrees curve
        
        self.vx = velocity_magnitude * math.cos(base_angle + curve_angle)
        self.vy = velocity_magnitude * math.sin(base_angle + curve_angle)
        
        # Calculate lift coefficient based on spin
        self.lift_coefficient = 0.25 * (1 - math.exp(-0.5 * abs(self.spin)))
        self.in_motion = True

    def update(self):
        if self.in_motion:
            # Calculate Magnus effect force
            velocity = math.hypot(self.vx, self.vy)
            if velocity > 0:
                magnus_force = 0.5 * 1.225 * (velocity**2) * self.lift_coefficient * math.pi * (self.radius**2)
                magnus_direction = -math.copysign(1, self.spin)  # Direction depends on spin
                
                # Apply Magnus force components
                self.vy += (GRAVITY - (magnus_force * magnus_direction)/BALL_MASS)
            
            # Apply air resistance
            self.vx *= AIR_RESISTANCE
            self.vy *= AIR_RESISTANCE
            
            # Update position
            self.x += self.vx
            self.y += self.vy
            
            # Boundary checks
            if self.x < 0 or self.x > SCREEN_WIDTH or self.y > SCREEN_HEIGHT:
                self.in_motion = False

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 2)

class Player:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.radius = 12
        self.color = color

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class KickingPlayer(Player):
    def __init__(self, x, y):
        super().__init__(x, y, ORANGE)
        self.kick_animation = False
        self.animation_frame = 0

    def start_kick(self):
        self.kick_animation = True
        self.animation_frame = 0

    def update(self):
        if self.kick_animation:
            self.animation_frame += 1
            if self.animation_frame >= 10:
                self.kick_animation = False

    def draw(self, screen):
        angle = 30 * math.sin(self.animation_frame * 0.3)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        # Draw kicking leg
        leg_length = 20
        leg_x = self.x + math.cos(math.radians(angle)) * leg_length
        leg_y = self.y + math.sin(math.radians(angle)) * leg_length
        pygame.draw.line(screen, BLACK, (self.x, self.y), (leg_x, leg_y), 3)

class Goalkeeper:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 20
        self.height = 30
        self.speed = 1.5
        self.direction = 1

    def update(self):
        self.x += self.speed * self.direction
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = (SCREEN_WIDTH + GOAL_WIDTH) // 2
        
        if self.x <= goal_left + self.width//2 or self.x >= goal_right - self.width//2:
            self.direction *= -1

    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, (self.x - self.width//2, self.y - self.height//2, self.width, self.height))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Realistic Free Kick Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        self.setup_game()

    def setup_game(self):
        # Initialize game objects
        self.ball = Ball(SCREEN_WIDTH//2 + random.randint(-100, 100), 
                        SCREEN_HEIGHT - 150)
        self.kicking_player = KickingPlayer(self.ball.x - 30, self.ball.y)
        self.goalkeeper = Goalkeeper(SCREEN_WIDTH//2, 50)
        self.wall_players = self.create_wall()
        self.shoot_requested = False

    def create_wall(self):
        wall_players = []
        wall_size = random.randint(2, 6)
        wall_distance = max(120, random.randint(120, 180))
        wall_y = self.ball.y - wall_distance
        wall_width = wall_size * 30
        start_x = self.ball.x - wall_width // 2
        
        for i in range(wall_size):
            player_x = start_x + i * 30 + random.randint(-5, 5)
            player_y = wall_y + random.randint(-10, 10)
            wall_players.append(Player(player_x, player_y, BLUE))
        return wall_players

    def handle_collisions(self):
        # Wall collisions
        for player in self.wall_players:
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            distance = math.hypot(dx, dy)
            
            if distance < self.ball.radius + player.radius:
                self.ball.in_motion = False
                return True
        
        # Goalkeeper collision
        gk_dx = self.ball.x - self.goalkeeper.x
        gk_dy = self.ball.y - self.goalkeeper.y
        if abs(gk_dx) < 20 and abs(gk_dy) < 30:
            self.ball.in_motion = False
            return True
        
        return False

    def check_goal(self):
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = (SCREEN_WIDTH + GOAL_WIDTH) // 2
        return (goal_left <= self.ball.x <= goal_right and 
                50 <= self.ball.y <= 50 + GOAL_HEIGHT)

    def draw_field(self):
        self.screen.fill(GREEN)
        # Draw goal
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        pygame.draw.rect(self.screen, WHITE, (goal_left, 50, GOAL_WIDTH, GOAL_HEIGHT), 3)
        # Draw penalty area
        penalty_left = (SCREEN_WIDTH - 165) // 2
        pygame.draw.rect(self.screen, WHITE, (penalty_left, SCREEN_HEIGHT-180, 165, 180), 3)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.ball.in_motion:
                        self.kicking_player.start_kick()
                        self.shoot_requested = True
                    
                    if event.key == pygame.K_UP:
                        self.power = min(100, self.power + 5)
                    if event.key == pygame.K_DOWN:
                        self.power = max(0, self.power - 5)

            # Update game state
            self.kicking_player.update()
            self.ball.update()
            self.goalkeeper.update()

            # Handle shooting
            if self.shoot_requested and not self.kicking_player.kick_animation:
                cursor_pos = pygame.mouse.get_pos()
                self.ball.shoot(self.power, *cursor_pos)
                self.shoot_requested = False

            # Check game state
            if self.ball.in_motion:
                if self.handle_collisions() or not self.ball.in_motion:
                    self.setup_game()
                elif self.check_goal():
                    self.setup_game()

            # Drawing
            self.draw_field()
            for player in self.wall_players:
                player.draw(self.screen)
            self.goalkeeper.draw(self.screen)
            self.ball.draw(self.screen)
            self.kicking_player.draw(self.screen)
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

GOAL_WIDTH = int(9.5 * PIXELS_PER_METER)  # 9.5 meters
GOAL_HEIGHT = int(3.2 * PIXELS_PER_METER)  # 3.2 meters

if __name__ == "__main__":
    game = Game()
    game.run()
