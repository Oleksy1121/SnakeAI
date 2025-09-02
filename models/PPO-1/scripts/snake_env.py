import sys
import math
import random
import logging
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# Colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)


class SnakeEnv(gym.Env):
    """
    Snake environment for Gymnasium.
    
    Features:
    - Action space: Discrete(3) [straight, right, left]
    - Observation space: 
        - image: grid with snake and food
        - vector: numerical features (distance, direction, etc.)
    - Supports rendering with Pygame.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}
    CELL_SIZE = 10

    def __init__(self, verbose: int = 0, max_apples: int = 1, initial_length: int = 3) -> None:
        """
        Initialize the Snake environment.
        
        Args:
            verbose (int): verbosity level (0 = silent, >0 = debug messages).
            max_apples (int): maximum number of apples on the board at once.
        """
        super().__init__()
        self.verbose = verbose
        self.max_apples = max_apples
        self.initial_length = initial_length

        # Board dimensions
        self.frame_size_x = 360
        self.frame_size_y = 360
        self.grid_width = self.frame_size_x // self.CELL_SIZE
        self.grid_height = self.frame_size_y // self.CELL_SIZE

        # Action space: 0 = straight, 1 = right, 2 = left
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.grid_height, self.grid_width, 1),
                    dtype=np.uint8,
                ),
                "vector": spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32),
            }
        )

        self.difficulty = 4

        # Game state
        self.game_initialized = False
        self.game_window = None
        self.fps_controller = None
        self.previous_snake_pos = None
        self.best_distance_to_apple = None

    # ----------------------
    #  Initialization
    # ----------------------
    def _init_pygame(self) -> None:
        """Initialize the Pygame window and game loop."""
        if not self.game_initialized:
            check_errors = pygame.init()
            if check_errors[1] > 0:
                logging.error("Pygame init error: %s errors found", check_errors[1])
                sys.exit(-1)

            pygame.display.set_caption("Snake Eater")
            self.game_window = pygame.display.set_mode(
                (self.frame_size_x, self.frame_size_y)
            )
            self.fps_controller = pygame.time.Clock()
            self.game_initialized = True
            logging.info("Pygame initialized.")

    # ----------------------
    #  Environment logic
    # ----------------------
    def step(self, action: int):
        """
        Perform a single step in the environment.
        
        Args:
            action (int): action index (0=straight, 1=right, 2=left).
        
        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        self.reward = 0
        self.terminated = False
        self.truncated = False

        self.frame_iteration += 1
        self.previous_snake_pos = list(self.snake_pos)

        # Map direction to new orientation
        direction_map = {
            "UP": {"straight": "UP", "right": "RIGHT", "left": "LEFT"},
            "DOWN": {"straight": "DOWN", "right": "LEFT", "left": "RIGHT"},
            "LEFT": {"straight": "LEFT", "right": "UP", "left": "DOWN"},
            "RIGHT": {"straight": "RIGHT", "right": "DOWN", "left": "UP"},
        }
        turn_actions = ["straight", "right", "left"]
        self.direction = direction_map[self.direction][turn_actions[action]]

        # Move snake
        new_head_pos = list(self.snake_pos)
        if self.direction == "UP":
            new_head_pos[1] -= self.CELL_SIZE
        elif self.direction == "DOWN":
            new_head_pos[1] += self.CELL_SIZE
        elif self.direction == "LEFT":
            new_head_pos[0] -= self.CELL_SIZE
        elif self.direction == "RIGHT":
            new_head_pos[0] += self.CELL_SIZE

        self.snake_pos = new_head_pos
        self.snake_body.insert(0, list(self.snake_pos))

        # Rewards and penalties
        self.reward += self._calculate_apple_reward()
        self.reward += self._calculate_wall_penalty()
        self.reward += 0.01  # small living reward

        # Check apple collision
        ate_apple = self._check_and_handle_apple()

        if not ate_apple:
            self.snake_body.pop()

        # Game over conditions
        if self._is_game_over():
            self.reward -= 1
            self.terminated = True

        if self.frame_iteration > 1000 + len(self.snake_body):
            self.reward -= 1
            self.truncated = True

        self.observation = self._get_observation()
        self.info = {"score": self.score, "snake_length": len(self.snake_body)}
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed (int, optional): random seed.
            options (dict, optional): additional options.
        
        Returns:
            tuple: observation, info
        """
        super().reset(seed=seed)
        self.frame_iteration = 0

        # Initial snake
        self.snake_pos = [100, 50]
        self.previous_snake_pos = list(self.snake_pos)
        self.snake_body = []
        for i in range(self.initial_length):
            self.snake_body.append([self.snake_pos[0] - i * self.CELL_SIZE, self.snake_pos[1]])

        # Spawn apples
        self.food_positions = []
        while len(self.food_positions) < self.max_apples:
            pos = [
                random.randrange(0, self.grid_width) * self.CELL_SIZE,
                random.randrange(0, self.grid_height) * self.CELL_SIZE,
            ]
            if pos not in self.snake_body and pos not in self.food_positions:
                self.food_positions.append(pos)

        self.direction = "RIGHT"
        self.score = 0

        self.observation = self._get_observation()
        self.info = {"score": self.score, "snake_length": len(self.snake_body)}
        return self.observation, self.info

    def _get_observation(self):
        """
        Build the observation as a dict with:
        - "image": (H, W, 1) uint8 grid where
            0 = background, 85 = head, 170 = body, 255 = apple
        - "vector": 8-dim float32 features:
            [scaled_manhattan_to_nearest_apple,
            cos_angle_to_apple (0..1),
            sin_angle_to_apple (0..1),
            normalized_snake_length,
            dir_UP, dir_RIGHT, dir_DOWN, dir_LEFT]
        """
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        head_grid_x = self.snake_pos[0] // self.CELL_SIZE
        head_grid_y = self.snake_pos[1] // self.CELL_SIZE
        is_head_in_grid = 0 <= head_grid_x < self.grid_width and 0 <= head_grid_y < self.grid_height

        if is_head_in_grid:
            grid[head_grid_y, head_grid_x] = 85  # head

        # Apples
        for food_pos in self.food_positions:
            fx = food_pos[0] // self.CELL_SIZE
            fy = food_pos[1] // self.CELL_SIZE
            if 0 <= fx < self.grid_width and 0 <= fy < self.grid_height:
                grid[fy, fx] = 255  # apple

        # Snake body (excluding head)
        for seg in self.snake_body[1:]:
            sx = seg[0] // self.CELL_SIZE
            sy = seg[1] // self.CELL_SIZE
            if 0 <= sx < self.grid_width and 0 <= sy < self.grid_height:
                grid[sy, sx] = 170  # body

        # Vector features
        if self.food_positions and is_head_in_grid:
            # Nearest apple by Manhattan distance (in pixels)
            nearest_food = min(
                self.food_positions,
                key=lambda food: abs(food[0] - self.snake_pos[0]) + abs(food[1] - self.snake_pos[1]),
            )

            # Physical deltas (in pixels)
            dx_phys = nearest_food[0] - self.snake_pos[0]
            dy_phys = nearest_food[1] - self.snake_pos[1]

            # Angle to apple (normalize cos/sin to [0, 1] for stability)
            angle = math.atan2(dy_phys, dx_phys)
            cos_angle = (math.cos(angle) + 1) / 2.0
            sin_angle = (math.sin(angle) + 1) / 2.0

            # Direction one-hot
            dir_map = {
                "UP":    [1.0, 0.0, 0.0, 0.0],
                "RIGHT": [0.0, 1.0, 0.0, 0.0],
                "DOWN":  [0.0, 0.0, 1.0, 0.0],
                "LEFT":  [0.0, 0.0, 0.0, 1.0],
            }

            dir_to_vec = {
                "UP":    (0, -1),
                "DOWN":  (0, 1),
                "LEFT":  (-1, 0),
                "RIGHT": (1, 0),
            }
                        

            direction_onehot = dir_map[self.direction]

            # Manhattan distance on the grid
            food_grid_x = nearest_food[0] // self.CELL_SIZE
            food_grid_y = nearest_food[1] // self.CELL_SIZE
            current_distance = abs(food_grid_x - head_grid_x) + abs(food_grid_y - head_grid_y)
            max_possible_distance = (self.grid_width - 1) + (self.grid_height - 1)
            scaled_distance = current_distance / max_possible_distance

            snake_length = len(self.snake_body)


            # --- scan environment relative to head ---
            head_x, head_y = self.snake_pos
            forward_vec = dir_to_vec[self.direction]
            right_vec = (-forward_vec[1], forward_vec[0])
            left_vec = (forward_vec[1], -forward_vec[0])

            f_type, f_dist = self._scan_direction([head_x, head_y], forward_vec)
            l_type, l_dist = self._scan_direction([head_x, head_y], left_vec)
            r_type, r_dist = self._scan_direction([head_x, head_y], right_vec)

            # normalizacja dystansu (max = szerokość+wysokość w kratkach)
            max_dist = max(self.grid_width, self.grid_height)
            f_dist /= max_dist
            l_dist /= max_dist
            r_dist /= max_dist

            type_to_onehot = lambda t: [1.0 if t == i else 0.0 for i in range(1, 4)]
            f_type_oh = type_to_onehot(f_type)
            l_type_oh = type_to_onehot(l_type)
            r_type_oh = type_to_onehot(r_type)


            vector_obs = np.array(
                [
                    scaled_distance,
                    cos_angle,
                    sin_angle,
                    snake_length / (self.grid_width * self.grid_height),
                    *direction_onehot,
                    *f_type_oh, f_dist,
                    *l_type_oh, l_dist,
                    *r_type_oh, r_dist,
                ],
                dtype=np.float32,
            )
        else:
            # No apples or head out of grid -> zero vector
            vector_obs = np.zeros(8, dtype=np.float32)

        return {
            "image": np.expand_dims(grid, axis=-1),
            "vector": vector_obs,
        }


    def _calculate_apple_reward(self):
        """
        Shaped reward based on Manhattan distance to the nearest apple.
        - Moved closer: +0.05
        - Moved away:   -0.05
        - New best distance in this episode: +0.05
        """
        if not self.food_positions or self.previous_snake_pos is None:
            return 0.0

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        current_dist = min(manhattan(self.snake_pos, apple) for apple in self.food_positions)
        previous_dist = min(manhattan(self.previous_snake_pos, apple) for apple in self.food_positions)

        reward = 0.0
        if current_dist < previous_dist:
            reward += 0.05
            if self.verbose:
                logging.debug("Moved closer to apple: +0.05")
        elif current_dist > previous_dist:
            reward -= 0.05
            if self.verbose:
                logging.debug("Moved away from apple: -0.05")

        if self.best_distance_to_apple is None or current_dist < self.best_distance_to_apple:
            self.best_distance_to_apple = current_dist
            reward += 0.05
            if self.verbose:
                logging.debug("New best distance to apple: +0.05 (distance %s)", current_dist)

        return reward


    def _calculate_wall_penalty(self):
        """
        Penalty for moving closer to a wall when already within 5 cells.
        Each approach costs -0.5.
        """
        if self.previous_snake_pos is None:
            return 0.0

        def distances_to_walls(pos):
            x_cells = pos[0] // self.CELL_SIZE
            y_cells = pos[1] // self.CELL_SIZE
            left = x_cells
            right = (self.grid_width - 1) - x_cells
            top = y_cells
            bottom = (self.grid_height - 1) - y_cells
            return left, right, top, bottom

        penalty = 0.0
        c_left, c_right, c_top, c_bottom = distances_to_walls(self.snake_pos)
        p_left, p_right, p_top, p_bottom = distances_to_walls(self.previous_snake_pos)

        if c_left < 5 and c_left < p_left:
            penalty -= 0.5
            if self.verbose:
                logging.debug("Closer to left wall: -0.5")
        if c_right < 5 and c_right < p_right:
            penalty -= 0.5
            if self.verbose:
                logging.debug("Closer to right wall: -0.5")
        if c_top < 5 and c_top < p_top:
            penalty -= 0.5
            if self.verbose:
                logging.debug("Closer to top wall: -0.5")
        if c_bottom < 5 and c_bottom < p_bottom:
            penalty -= 0.5
            if self.verbose:
                logging.debug("Closer to bottom wall: -0.5")

        return penalty



    # ----------------------
    #  Rendering
    # ----------------------
    def render(self) -> None:
        """Render the game state with Pygame."""
        self._init_pygame()
        self.game_window.fill(BLACK)

        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(
                self.game_window, GREEN, pygame.Rect(pos[0], pos[1], self.CELL_SIZE, self.CELL_SIZE)
            )

        # Draw apples
        for food_pos in self.food_positions:
            pygame.draw.rect(
                self.game_window, WHITE, pygame.Rect(food_pos[0], food_pos[1], self.CELL_SIZE, self.CELL_SIZE)
            )

        # Show score
        self._show_score(WHITE, "consolas", 20)
        pygame.display.update()
        self.fps_controller.tick(self.difficulty)

    def _show_score(self, color, font, size):
        """Display the current score in the game window."""
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render(f"Score : {self.score}", True, color)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def close(self) -> None:
        """Close the environment and release resources."""
        if self.game_initialized:
            pygame.quit()
            self.game_initialized = False
            logging.info("Pygame closed.")

    # ----------------------
    #  Helpers
    # ----------------------
    def _check_and_handle_apple(self) -> bool:
        """Check if the snake eats an apple and handle apple respawn."""
        for food_pos in self.food_positions:
            if self.snake_pos == food_pos:
                self.score += 1
                self.reward += 1
                self.food_positions.remove(food_pos)
                self.best_distance_to_apple = None
                # Respawn apples
                while len(self.food_positions) < self.max_apples:
                    new_food_pos = [
                        random.randrange(0, self.grid_width) * self.CELL_SIZE,
                        random.randrange(0, self.grid_height) * self.CELL_SIZE,
                    ]
                    if new_food_pos not in self.snake_body and new_food_pos not in self.food_positions:
                        self.food_positions.append(new_food_pos)
                        break
                self.frame_iteration = 0
                return True
        return False

    def _is_game_over(self) -> bool:
        """Check collision with wall or snake body."""
        if (
            self.snake_pos[0] < 0
            or self.snake_pos[0] >= self.frame_size_x
            or self.snake_pos[1] < 0
            or self.snake_pos[1] >= self.frame_size_y
        ):
            return True

        if self.snake_pos in self.snake_body[1:]:
            return True
        return False

    def _scan_direction(self, start_pos, direction_vec):
        x, y = start_pos
        dx, dy = direction_vec
        steps = 0
        while True:
            x += dx * self.CELL_SIZE
            y += dy * self.CELL_SIZE
            steps += 1

            # check wall
            if x < 0 or x >= self.frame_size_x or y < 0 or y >= self.frame_size_y:
                return (1, steps)  # 1 = wall

            # check apples
            if [x, y] in self.food_positions:
                return (2, steps)  # 2 = apple

            # check snake body (excluding first 3 segments after head)
            if [x, y] in self.snake_body[3:]:
                return (3, steps)  # 3 = tail
