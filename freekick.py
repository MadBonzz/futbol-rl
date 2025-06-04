import pygame
import math
import random
import torch
import numpy as np

pygame.init()

# Constants based on FIFA field dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 12

FIELD_WIDTH = int(68 * PIXELS_PER_METER)
FIELD_HEIGHT = int(50 * PIXELS_PER_METER)
GOAL_WIDTH = int(9.5 * PIXELS_PER_METER)
GOAL_HEIGHT = int(3.2 * PIXELS_PER_METER)
PENALTY_AREA_WIDTH = int(40.32 * PIXELS_PER_METER)
PENALTY_AREA_HEIGHT = int(16.5 * PIXELS_PER_METER)
WALL_MIN_DISTANCE = int(9.15 * PIXELS_PER_METER)

PLAYER_HEIGHT = 1.8
GOALKEEPER_HEIGHT = 1.9
GOALPOST_HEIGHT = 2.44

GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)

class EventLogger:
    def __init__(self):
        self.reset_log()
    
    def reset_log(self):
        self.event_log = []
        print("Event log reset for new scenario")
    
    def log_key_press(self, key, wall_passed=0, goal_scored=0):
        event_tuple = (key, wall_passed, goal_scored)
        self.event_log.append(event_tuple)
        print(f"Logged event: {event_tuple}")
    
    def get_log(self):
        return self.event_log.copy()
    
    def print_final_log(self):
        print("\n" + "="*50)
        print("FINAL EVENT LOG:")
        print("="*50)
        for i, event in enumerate(self.event_log, 1):
            print(f"{i}. {event}")
        print("="*50)
        return self.event_log

class KickingPlayer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 12
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
        
        # Draw kicking leg based on animation frame
        if self.is_kicking:
            leg_offset = self.animation_frame * 3
            leg_x = self.x
            leg_y = self.y - leg_offset  # Kick towards goal (negative y)
            pygame.draw.line(screen, self.color, (self.x, self.y - 5), (leg_x, leg_y), 5)
        else:
            # Normal stance
            pygame.draw.line(screen, self.color, (self.x - 5, self.y - 5), (self.x - 5, self.y - 20), 5)
            pygame.draw.line(screen, self.color, (self.x + 5, self.y - 5), (self.x + 5, self.y - 20), 5)

class Player:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.radius = 12
        self.color = color
        self.height = PLAYER_HEIGHT * PIXELS_PER_METER
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Goalkeeper:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 15
        self.height = 25
        self.speed = 1.5
        self.direction = 1
        self.player_height = GOALKEEPER_HEIGHT * PIXELS_PER_METER
    
    def update(self):
        # Simple AI movement (left-right within goal area)
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
        self.x = x - 50
        self.y = y - 25
        self.width = 200
        self.height = 25
        self.power = 50
        self.max_power = 100
    
    def increase_power(self):
        self.power = min(self.max_power, self.power + 5)
    
    def decrease_power(self):
        self.power = max(0, self.power - 5)
    
    def draw(self, screen, font):
        pygame.draw.rect(screen, GRAY, (self.x, self.y, self.width, self.height))
        power_width = (self.power / self.max_power) * self.width
        color = YELLOW if self.power < 70 else RED
        pygame.draw.rect(screen, color, (self.x, self.y, power_width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        text = font.render(f"Power: {self.power}%", True, BLACK)
        screen.blit(text, (self.x - 80, self.y))

class ElevationBar:
    def __init__(self, x, y):
        self.x = x - 50
        self.y = y - 20
        self.width = 200
        self.height = 25
        self.elevation = 25
        self.max_elevation = 100
    
    def increase_elevation(self):
        self.elevation = min(self.max_elevation, self.elevation + 5)
    
    def decrease_elevation(self):
        self.elevation = max(0, self.elevation - 5)
    
    def get_elevation_factor(self):
        return self.elevation / self.max_elevation
    
    def get_elevation_angle(self):
        max_angle = math.pi / 3
        return self.get_elevation_factor() * max_angle
    
    def draw(self, screen, font):
        pygame.draw.rect(screen, GRAY, (self.x, self.y, self.width, self.height))
        elevation_width = (self.elevation / self.max_elevation) * self.width
        color = YELLOW if self.elevation < 60 else RED
        pygame.draw.rect(screen, color, (self.x, self.y, elevation_width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        text = font.render(f"Elevation: {self.elevation}%", True, BLACK)
        screen.blit(text, (self.x - 100, self.y))

class CurveBar:
    def __init__(self, x, y):
        self.x = x - 50
        self.y = y - 15
        self.width = 200
        self.height = 25
        self.curve = 50
        self.max_curve = 100
    
    def increase_curve(self):
        self.curve = min(self.max_curve, self.curve + 5)
    
    def decrease_curve(self):
        self.curve = max(0, self.curve - 5)
    
    def get_curve_factor(self):
        return (self.curve / 50.0) - 1.0
    
    def draw(self, screen, font):
        pygame.draw.rect(screen, GRAY, (self.x, self.y, self.width, self.height))
        center_x = self.x + self.width // 2
        pygame.draw.line(screen, BLACK, (center_x, self.y), (center_x, self.y + self.height), 2)
        indicator_x = self.x + (self.curve / 100.0) * self.width
        indicator_width = 8
        color = BLUE if self.curve < 45 else YELLOW if self.curve < 55 else RED
        pygame.draw.rect(screen, color, (indicator_x - indicator_width//2, self.y, indicator_width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        curve_text = "Left" if self.curve < 45 else "Straight" if self.curve < 55 else "Right"
        text = font.render(f"Curve: {curve_text}", True, BLACK)
        screen.blit(text, (self.x - 100, self.y))

class Ball:
    def __init__(self, x, y, z):
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
        self.gravity = 0.98
        self.bounce_damping = 0.6
        self.magnus_strength = 0.15
        self.in_motion = False
        self.curve_factor = 0
        self.trajectory_points = []
        self.max_trajectory_points = 1000
        self.wall_distance = 0
        self.goal_distance_x = 0
        self.goal_distance_y = 0
        self.final_position = None
        self.stopped_before_completion = False
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.in_motion = False
        self.curve_factor = 0
        self.trajectory_points = []
        self.final_position = None
        self.stopped_before_completion = False
    
    def shoot(self, power, elevation_angle, curve_factor):
        base_velocity = power * 0.3
        horizontal_velocity = base_velocity * math.cos(elevation_angle)
        vertical_velocity = base_velocity * math.sin(elevation_angle)
        
        self.vy = -horizontal_velocity
        self.vx = -(self.goal_distance_x / 90) * horizontal_velocity
        self.vz = vertical_velocity
        
        self.curve_factor = curve_factor
        self.trajectory_points = [(self.x, self.y, self.z)]
        self.in_motion = True
        self.stopped_before_completion = False
        
        print(f"Shot initiated: Power={power}%, Elevation={math.degrees(elevation_angle):.1f}°, Curve={curve_factor:.2f}")
        print(f"Initial velocities: vx={self.vx:.2f}, vy={self.vy:.2f}, vz={self.vz:.2f}")
    
    def update(self):
        if not self.in_motion:
            return False
        
        self.vz -= self.gravity
        
        if self.z > 0:
            speed_factor = abs(self.vy) / 2
            magnus_force = self.curve_factor * self.magnus_strength * speed_factor
            self.vx += magnus_force
        
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        
        if len(self.trajectory_points) < self.max_trajectory_points:
            self.trajectory_points.append((self.x, self.y, self.z))
        
        # Ground collision
        if self.z <= 0 and self.vz < 0:
            self.z = 0
            self.final_position = (self.x, self.y, self.z)
            self.in_motion = False
            return True
        
        # Check if ball has stopped moving
        if (abs(self.vx) < 0.05 and abs(self.vy) < 0.05 and 
            abs(self.vz) < 0.05 and self.z <= 0.1):
            self.final_position = (self.x, self.y, self.z)
            self.in_motion = False
            return True
        
        # Check boundaries
        if (self.x < -50 or self.x > SCREEN_WIDTH + 50 or 
            self.y < -50 or self.y > SCREEN_HEIGHT + 50):
            self.final_position = (self.x, self.y, self.z)
            return True
        
        return False
    
    def get_trajectory_at_y(self, target_y):
        if not self.trajectory_points:
            return None
            
        for i in range(1, len(self.trajectory_points)):
            x1, y1, z1 = self.trajectory_points[i-1]
            x2, y2, z2 = self.trajectory_points[i]
            
            if ((y1 >= target_y >= y2) or (y2 >= target_y >= y1)) and y1 != y2:
                t = (target_y - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                z = z1 + t * (z2 - z1)
                return (x, z)
        return None
    
    def get_final_y_position(self):
        if self.final_position:
            return self.final_position[1]
        elif self.trajectory_points:
            return self.trajectory_points[-1][1]
        else:
            return self.y
        
    def get_final_x_position(self):
        if self.final_position:
            return self.final_position[0]
        elif self.trajectory_points:
            return self.trajectory_points[-1][0]
        else:
            return self.x
    
    def draw(self, screen):
        # Draw shadow
        shadow_radius = max(3, int(self.radius * 0.8 - self.z / 40))
        pygame.draw.circle(screen, (50, 50, 50), (int(self.x), int(self.y)), shadow_radius)
        
        # Draw ball with height perspective
        ball_radius = max(6, int(self.radius - self.z / 50))
        ball_pos = (int(self.x), int(self.y - self.z / 10))
        pygame.draw.circle(screen, WHITE, ball_pos, ball_radius)
        pygame.draw.circle(screen, BLACK, ball_pos, ball_radius, 2)
        
        # Draw trajectory line
        if len(self.trajectory_points) > 1:
            for i in range(1, min(len(self.trajectory_points), 30)):
                prev_x, prev_y, prev_z = self.trajectory_points[i-1]
                curr_x, curr_y, curr_z = self.trajectory_points[i]
                prev_screen = (int(prev_x), int(prev_y - prev_z / 10))
                curr_screen = (int(curr_x), int(curr_y - curr_z / 10))
                pygame.draw.line(screen, (200, 200, 200), prev_screen, curr_screen, 1)

class EnhancedCollisionDetector:
    @staticmethod
    def comprehensive_analysis(ball, wall_players, goalkeeper):
        """
        Comprehensive analysis that determines wall_passed and goal_scored accurately
        """
        wall_passed = 0
        goal_scored = 0
        analysis_result = "No shot taken"
        
        if not ball.trajectory_points or len(ball.trajectory_points) < 2:
            return wall_passed, goal_scored, "No trajectory data"
        
        # Get key positions
        wall_y = wall_players[0].y if wall_players else None
        goal_y = 50
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_x = goal_left - (GOAL_WIDTH / 2)
        ball_final_y = ball.get_final_y_position()
        ball_final_x = ball.get_final_x_position()
        final_dist = np.sqrt(np.square(goal_x - ball_final_x) + (np.square(goal_y - ball_final_y)))
        
        print(f"Analysis - Ball final Y: {ball_final_y}, Wall Y: {wall_y}, Goal Y: {goal_y}")
        
        # Check if ball reached or passed the wall
        if wall_y is not None:
            if ball_final_y <= wall_y:  # Ball reached or passed wall line
                # Check if ball actually hit wall or passed over/around it
                wall_collision, wall_details = EnhancedCollisionDetector.check_wall_collision(ball, wall_players)
                if wall_collision:
                    analysis_result = f"Wall collision: {wall_details}"
                    # Ball hit wall, didn't pass
                    wall_passed = 0
                else:
                    # Ball passed wall (over or around)
                    wall_passed = 1
                    analysis_result = "Ball passed wall"
                    
                    # If wall was passed, check goal
                    if ball_final_y <= goal_y:  # Ball reached goal line
                        is_goal, goal_details = EnhancedCollisionDetector.check_goal_collision(ball)
                        if is_goal:
                            goal_scored = 1
                            analysis_result = f"GOAL! {goal_details}"
                        else:
                            # Check goalkeeper save
                            gk_collision, gk_details = EnhancedCollisionDetector.check_goalkeeper_collision(ball, goalkeeper)
                            if gk_collision:
                                analysis_result = f"Goalkeeper save: {gk_details}"
                            else:
                                analysis_result = f"Goal miss: {goal_details}"
                    else:
                        analysis_result = "Ball stopped between wall and goal"
            else:
                # Ball stopped before reaching wall
                analysis_result = f"Ball stopped before wall (final Y: {ball_final_y:.1f}, wall Y: {wall_y:.1f})"
                wall_passed = 0
        else:
            # No wall present
            wall_passed = 1  # Consider wall as passed if no wall exists
            if ball_final_y <= goal_y:
                is_goal, goal_details = EnhancedCollisionDetector.check_goal_collision(ball)
                if is_goal:
                    goal_scored = 1
                    analysis_result = f"GOAL! {goal_details}"
                else:
                    gk_collision, gk_details = EnhancedCollisionDetector.check_goalkeeper_collision(ball, goalkeeper)
                    if gk_collision:
                        analysis_result = f"Goalkeeper save: {gk_details}"
                    else:
                        analysis_result = f"Goal miss: {goal_details}"
        
        print(f"Final analysis: wall_passed={wall_passed}, goal_scored={goal_scored}, result='{analysis_result}'")
        return final_dist, wall_passed, goal_scored, analysis_result
    
    @staticmethod
    def check_wall_collision(ball, wall_players):
        if not wall_players or not ball.trajectory_points:
            return False, "No wall data"
        
        wall_y = wall_players[0].y
        ball_position = ball.get_trajectory_at_y(wall_y)
        
        if ball_position is None:
            return False, "Ball never reaches wall line"
        
        ball_x, ball_z = ball_position
        ball_height_meters = ball_z / PIXELS_PER_METER
        
        for player in wall_players:
            horizontal_distance = abs(ball_x - player.x)
            if horizontal_distance < (ball.radius + player.radius):
                if ball_height_meters <= PLAYER_HEIGHT:
                    return True, f"Ball hits player at height {ball_height_meters:.2f}m"
                else:
                    print(f"Ball passes over player! Height: {ball_height_meters:.2f}m > {PLAYER_HEIGHT}m")
        
        return False, f"Ball passes around wall at height {ball_height_meters:.2f}m"
    
    @staticmethod
    def check_goal_collision(ball):
        if not ball.trajectory_points:
            return False, "No trajectory data"
        
        goal_y = 50
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        
        ball_position = ball.get_trajectory_at_y(goal_y)
        if ball_position is None:
            return False, "Ball never reaches goal line"
        
        ball_x, ball_z = ball_position
        ball_height_meters = ball_z / PIXELS_PER_METER
        
        within_width = goal_left <= ball_x <= goal_right
        within_height = ball_height_meters <= GOALPOST_HEIGHT and ball_z >= 0
        
        if within_width and within_height:
            return True, f"Ball enters at height {ball_height_meters:.2f}m"
        
        if not within_width:
            if ball_x < goal_left:
                miss_distance = (goal_left - ball_x) / PIXELS_PER_METER
                return False, f"Wide left by {miss_distance:.2f}m"
            else:
                miss_distance = (ball_x - goal_right) / PIXELS_PER_METER
                return False, f"Wide right by {miss_distance:.2f}m"
        
        if not within_height:
            if ball_height_meters > GOALPOST_HEIGHT:
                excess_height = ball_height_meters - GOALPOST_HEIGHT
                return False, f"Over crossbar by {excess_height:.2f}m"
            else:
                return False, f"Under crossbar (ground shot)"
        
        return False, "Miss (unknown reason)"
    
    @staticmethod
    def check_goalkeeper_collision(ball, goalkeeper):
        if not ball.trajectory_points:
            return False, "No trajectory data"
        
        ball_position = ball.get_trajectory_at_y(goalkeeper.y)
        if ball_position is None:
            return False, "Ball never reaches goalkeeper"
        
        ball_x, ball_z = ball_position
        ball_height_meters = ball_z / PIXELS_PER_METER
        horizontal_distance = abs(ball_x - goalkeeper.x)
        
        if horizontal_distance < (ball.radius + goalkeeper.width//2):
            if ball_height_meters <= GOALKEEPER_HEIGHT:
                return True, f"Goalkeeper saves at height {ball_height_meters:.2f}m"
            else:
                print(f"Ball passes over goalkeeper! Height: {ball_height_meters:.2f}m")
        
        return False, "Ball passes by goalkeeper"

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Enhanced Football Free Kick - Improved Logging")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.big_font = pygame.font.Font(None, 32)
        
        self.event_logger = EventLogger()
        self.setup_scenario()
        self.collision_detector = EnhancedCollisionDetector()
        self.score = 0
        self.attempts = 0
        self.last_result = ""
        self.show_debug = True

    def setup_scenario(self):
        # Reset logger for new scenario
        self.event_logger.reset_log()

        self.power_bar = PowerBar(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 80)
        self.elevation_bar = ElevationBar(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 60)
        self.curve_bar = CurveBar(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 40)
        
        # Ball positioning
        ball_y = SCREEN_HEIGHT - random.randint(100, 150)
        ball_x = SCREEN_WIDTH // 2 + random.randint(-90, 90)
        self.ball = Ball(ball_x, ball_y, 0)
        
        # Kicking player
        self.kicking_player = KickingPlayer(ball_x, ball_y + 25)
        
        # Goalkeeper
        self.goalkeeper = Goalkeeper(SCREEN_WIDTH // 2, 50)
        
        # Wall setup
        wall_size = random.randint(3, 5)
        wall_y = ball_y - random.randint(120, 180)
        
        self.wall_players = []
        wall_spacing = 30
        wall_start_x = ball_x - (wall_size * wall_spacing) // 2
        
        for i in range(wall_size):
            player_x = wall_start_x + i * wall_spacing + random.randint(-5, 5)
            self.wall_players.append(Player(player_x, wall_y, BLUE))
        
        # Store distances
        self.ball.wall_distance = abs(ball_y - wall_y)
        self.ball.goal_distance_x = ball_x - SCREEN_WIDTH // 2
        self.ball.goal_distance_y = abs(ball_y - 50)
        
        print(f"New scenario: Wall distance: {self.ball.wall_distance/PIXELS_PER_METER:.1f}m, "
              f"Goal distance: {self.ball.goal_distance_y/PIXELS_PER_METER:.1f}m")
    
    def get_frame_as_tensor(self):
        surface = pygame.display.get_surface()
        
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = np.transpose(frame_array, (1, 0, 2))
        
        frame_tensor = torch.from_numpy(frame_array).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor

    def draw_field(self):
        self.screen.fill(GREEN)
        
        # Goal area
        goal_left = (SCREEN_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        
        # Goal posts
        pygame.draw.rect(self.screen, WHITE, (goal_left - 5, 30, 10, GOAL_HEIGHT))
        pygame.draw.rect(self.screen, WHITE, (goal_right - 5, 30, 10, GOAL_HEIGHT))
        
        # Goal crossbar
        pygame.draw.line(self.screen, WHITE, (goal_left, 30), (goal_right, 30), 5)
        
        # Goal net pattern
        for i in range(goal_left, goal_right, 15):
            pygame.draw.line(self.screen, WHITE, (i, 30), (i, 30 + GOAL_HEIGHT), 1)
        
        # Penalty area
        penalty_left = (SCREEN_WIDTH - PENALTY_AREA_WIDTH) // 2
        pygame.draw.rect(self.screen, WHITE, 
                        (penalty_left, 30, PENALTY_AREA_WIDTH, PENALTY_AREA_HEIGHT), 3)
        
        # Center line
        center_y = SCREEN_HEIGHT // 2
        pygame.draw.line(self.screen, WHITE, (0, center_y), (SCREEN_WIDTH, center_y), 3)
        
        # Field boundaries
        pygame.draw.rect(self.screen, WHITE, (10, 30, SCREEN_WIDTH-20, SCREEN_HEIGHT-60), 3)

    def draw_game_objects(self):
        for player in sorted(self.wall_players, key=lambda p: p.y):
            player.draw(self.screen)
        
        self.kicking_player.draw(self.screen)
        self.goalkeeper.draw(self.screen)
        self.ball.draw(self.screen)

    def draw_ui(self):
        # Control bars
        self.power_bar.draw(self.screen, self.font)
        self.elevation_bar.draw(self.screen, self.font)
        self.curve_bar.draw(self.screen, self.font)
        
        # Score
        score_text = self.big_font.render(f"Goals: {self.score}/{self.attempts}", True, BLACK)
        self.screen.blit(score_text, (SCREEN_WIDTH-200, 10))
        
        # Result
        if self.last_result:
            result_color = GREEN if "GOAL" in self.last_result else RED
            result_text = self.font.render(self.last_result, True, result_color)
            result_rect = result_text.get_rect(center=(SCREEN_WIDTH//2, 80))
            self.screen.blit(result_text, result_rect)
        
        # Controls
        tutorial = [
            "CONTROLS:",
            "SPACE - Shoot",
            "↑/↓ - Power", 
            "W/S - Elevation",
            "A/D - Curve",
            "N - New scenario",
            "L - Show log"
        ]
        
        for i, line in enumerate(tutorial):
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (10, 350 + i*20))    

    def handle_shot_result(self):
        final_dist, wall_passed, goal_scored, analysis_result = self.collision_detector.comprehensive_analysis(
            self.ball, self.wall_players, self.goalkeeper
        )
        
        self.event_logger.log_key_press("Spacebar", wall_passed, goal_scored)
        
        if goal_scored:
            self.score += 1
        
        self.last_result = analysis_result
        return final_dist, wall_passed, goal_scored

    def step(self, action):
        if action == 0:
            self.power_bar.increase_power()
        elif action == 1:
            self.power_bar.decrease_power()
        elif action == 2:
            self.elevation_bar.increase_elevation()
        elif action == 3:
            self.elevation_bar.decrease_elevation()
        elif action == 4:
            self.curve_bar.increase_curve()
        elif action == 5:
            self.curve_bar.decrease_curve()
        elif action == 6:
            power = self.power_bar.power
            elevation_angle = self.elevation_bar.get_elevation_angle()
            curve_factor = self.curve_bar.get_curve_factor()
            self.ball.shoot(power, elevation_angle, curve_factor)
            self.kicking_player.start_kick()
            self.attempts += 1

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Print final log before quitting
                    self.event_logger.print_final_log()
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.ball.in_motion:
                        power = self.power_bar.power
                        elevation_angle = self.elevation_bar.get_elevation_angle()
                        curve_factor = self.curve_bar.get_curve_factor()
                        self.ball.shoot(power, elevation_angle, curve_factor)
                        self.kicking_player.start_kick()
                        self.attempts += 1
                        # Note: Spacebar event will be logged in handle_shot_result with results
                    elif event.key == pygame.K_UP:
                        self.power_bar.increase_power()
                        self.event_logger.log_key_press("UP", 0, 0)
                    elif event.key == pygame.K_DOWN:
                        self.power_bar.decrease_power()
                        self.event_logger.log_key_press("DOWN", 0, 0)
                    elif event.key == pygame.K_w:
                        self.elevation_bar.increase_elevation()
                        self.event_logger.log_key_press("W", 0, 0)
                    elif event.key == pygame.K_s:
                        self.elevation_bar.decrease_elevation()
                        self.event_logger.log_key_press("S", 0, 0)
                    elif event.key == pygame.K_a:
                        self.curve_bar.decrease_curve()
                        self.event_logger.log_key_press("A", 0, 0)
                    elif event.key == pygame.K_d:
                        self.curve_bar.increase_curve()
                        self.event_logger.log_key_press("D", 0, 0)
                    elif event.key == pygame.K_n:
                        # Print current log before new scenario
                        self.event_logger.print_final_log()
                        self.setup_scenario()
                        self.last_result = ""
                        self.event_logger.log_key_press("N", 0, 0)
                    elif event.key == pygame.K_l:
                        # Show current log
                        self.event_logger.print_final_log()
                        self.event_logger.log_key_press("L", 0, 0)
            
            # Update game objects
            ball_finished = self.ball.update()
            self.kicking_player.update()
            self.goalkeeper.update()
            
            # Handle shot completion
            if ball_finished and self.ball.trajectory_points:
                self.handle_shot_result()
                pygame.time.wait(1500)
                self.setup_scenario()
                self.last_result = ""
            
            # Render everything
            self.draw_field()
            self.draw_game_objects()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

game = Game()
game.run()