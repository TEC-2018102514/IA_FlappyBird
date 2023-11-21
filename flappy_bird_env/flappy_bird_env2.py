import gym
from gym import spaces
import pygame
import numpy as np

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        self.screen_width = 800
        self.screen_height = 576
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)  # 0: NOOP, 1: flap wings

        self.game = None
        self.clock = None

    def _reset(self):
        self.game = FlappyBirdGame(self.screen_width, self.screen_height)
        self.clock = pygame.time.Clock()
        return self.game.get_screen()

    def _step(self, action):
        if action == 1:
            self.game.flap_wings()

        reward, done = self.game.update()
        state = self.game.get_screen()
        return state, reward, done, {}

    def _render(self, mode="human", close=False):
        if mode == "rgb_array":
            return self.game.get_screen()
        elif mode == "human":
            self.game.render()
            pygame.display.flip()
            self.clock.tick(30)

    def _close(self):
        pygame.quit()


class FlappyBirdGame:
    def __init__(self, screen_width, screen_height):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.bird = Bird(self.screen)
        self.background = Background(self.screen)
        self.pipes = [Pipe(self.screen, self.screen_width + i * 300) for i in range(2)]
        self.base = Base(self.screen)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.bird.update()

        # Check for collisions
        if self.bird.check_collision(self.pipes):
            self._reset()
            return 0, True

        # Check for passing pipes
        for pipe in self.pipes:
            if pipe.rect.right < self.bird.x:
                self.bird.pass_pipe()
                pipe.passed = True

        # Check if bird hit the ground
        if self.bird.y + self.bird.image.get_height() >= 730:
            self._reset()
            return 0, True

        # Move pipes
        for pipe in self.pipes:
            pipe.update()

        # Generate new pipes
        if self.pipes[-1].rect.centerx < self.screen_width - 300:
            self.pipes.append(Pipe(self.screen, self.screen_width))

        self.base.update()

        # Calculate reward
        reward = 1 if self.pipes[0].rect.right < self.bird.x < self.pipes[0].rect.right + 5 else 0.001

        return reward, False

    def render(self):
        self.background.draw()
        for pipe in self.pipes:
            pipe.draw()
        self.base.draw()
        self.bird.draw()

    def _reset(self):
        self.bird.reset()
        for pipe in self.pipes:
            pipe.reset()

    def get_screen(self):
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(image_data, (1, 0, 2))


class Bird:
    def __init__(self, screen):
        self.screen = screen
        self.x = 100
        self.y = 350
        self.vel_y = 0
        self.tick_count = 0
        self.jump_count = 10
        self.image = pygame.image.load("flappy_bird_env\images\midflap.png")
        self.rect = self.image.get_rect()

    def jump(self):
        self.vel_y = -10.5
        self.tick_count = 0
        self.jump_count = 10

    def update(self):
        self.tick_count += 1

        displacement = self.vel_y * self.tick_count + 1.5 * self.tick_count ** 2

        if displacement >= 16:
            displacement = 16

        if displacement < 0:
            displacement -= 2

        self.y = max(0, min(self.y + displacement, 576 - self.image.get_height()))

        if displacement < 0 or self.y < self.y + 50:
            if self.jump_count > 0:
                self.jump()
                self.jump_count -= 1

    def draw(self):
        self.screen.blit(self.image, (self.x, self.y))

    def pass_pipe(self):
        pass  # Optional: Implement pass_pipe logic


class Pipe:
    def __init__(self, screen, x):
        self.screen = screen
        self.x = x
        self.height = 0
        self.gap = 100
        self.vel_x = -5
        self.image_top = pygame.image.load("flappy_bird_env\images\pipe.png")
        self.image_bottom = pygame.image.load("flappy_bird_env\images\pipe.png")
        self.passed = False

        self._set_height()

    def _set_height(self):
        self.height = np.random.randint(50, 300)
        self.image_top = pygame.transform.scale(self.image_top, (52, self.height))
        self.image_bottom = pygame.transform.scale(self.image_bottom, (52, 576 - self.height - self.gap))

    def update(self):
        self.x += self.vel_x

    def draw(self):
        self.screen.blit(self.image_top, (self.x, 0))
        self.screen.blit(self.image_bottom, (self.x, self.height + self.gap))

    def reset(self):
        self.x = self.screen.get_width() + 200
        self.passed = False
        self._set_height()


class Base:
    def __init__(self, screen):
        self.screen = screen
        self.vel_x = -5
        self.image = pygame.image.load("flappy_bird_env\images\base.png")
        self.x1 = 0
        self.x2 = self.image.get_width()

    def update(self):
        self.x1 += self.vel_x
        self.x2 += self.vel_x

        if self.x1 + self.image.get_width() < 0:
            self.x1 = self.x2 + self.image.get_width()

        if self.x2 + self.image.get_width() < 0:
            self.x2 = self.x1 + self.image.get_width()

    def draw(self):
        self.screen.blit(self.image, (self.x1, 576 - self.image.get_height()))
        self.screen.blit(self.image, (self.x2, 576 - self.image.get_height()))
