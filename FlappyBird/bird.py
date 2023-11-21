"""
This script defines two classes: Bird and BirdCollection, representing birds and a collection of birds in a game using Pygame.
"""

import pygame
import random
from defs import *
from nnet import Nnet
import numpy as np

class Bird:
    """
    The Bird class represents an individual bird in the game.
    """

    def __init__(self, gameDisplay):
        """
        Initializes a bird with its properties.

        Parameters:
            gameDisplay (pygame.Surface): The game display surface.
        """
        self.gameDisplay = gameDisplay
        self.state = BIRD_ALIVE
        self.img = pygame.image.load(BIRD_FILENAME)
        self.rect = self.img.get_rect()
        self.speed = 0
        self.fitness = 0
        self.time_lived = 0
        self.nnet = Nnet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)
        self.set_position(BIRD_START_X, BIRD_START_Y)

    def reset(self):
        """
        Resets the bird's properties to their initial state.
        """
        self.state = BIRD_ALIVE
        self.speed = 0
        self.fitness = 0
        self.time_lived = 0
        self.set_position(BIRD_START_X, BIRD_START_Y)

    def set_position(self, x, y):
        """
        Sets the position of the bird.

        Parameters:
            x (int): The new x-coordinate.
            y (int): The new y-coordinate.
        """
        self.rect.centerx = x
        self.rect.centery = y

    def move(self, dt):
        """
        Moves the bird based on its speed and gravity.

        Parameters:
            dt (int): The time elapsed since the last frame.
        """
        distance = 0
        new_speed = 0

        distance = (self.speed * dt) + (0.5 * GRAVITY * dt * dt)
        new_speed = self.speed + (GRAVITY * dt)

        self.rect.centery += distance
        self.speed = new_speed

        if self.rect.top < 0:
            self.rect.top = 0
            self.speed = 0

    def jump(self, pipes):
        """
        Makes the bird jump based on the neural network output.

        Parameters:
            pipes (list): List of pipes in the game.
        """
        inputs = self.get_inputs(pipes)
        val = self.nnet.get_max_value(inputs)
        if val > JUMP_CHANCE:
            self.speed = BIRD_START_SPEED

    def draw(self):
        """
        Draws the bird on the game display.
        """
        self.gameDisplay.blit(self.img, self.rect)

    def check_status(self, pipes):
        """
        Checks the status of the bird, updating its state and fitness.

        Parameters:
            pipes (list): List of pipes in the game.
        """
        if self.rect.bottom > DISPLAY_H:
            self.state = BIRD_DEAD
        else:
            self.check_hits(pipes)

    def assign_collision_fitness(self, p):
        """
        Assigns fitness to the bird based on collision with a pipe.

        Parameters:
            p (Pipe): The pipe with which the collision occurred.
        """
        gap_y = 0
        if p.pipe_type == PIPE_UPPER:
            gap_y = p.rect.bottom + PIPE_GAP_SIZE / 2
        else:
            gap_y = p.rect.top - PIPE_GAP_SIZE / 2

        self.fitness = -(abs(self.rect.centery - gap_y))

    def check_hits(self, pipes):
        """
        Checks if the bird collides with any pipes.

        Parameters:
            pipes (list): List of pipes in the game.
        """
        for p in pipes:
            if p.rect.colliderect(self.rect):
                self.state = BIRD_DEAD
                self.assign_collision_fitness(p)
                break

    def update(self, dt, pipes):
        """
        Updates the bird's position, state, and other properties.

        Parameters:
            dt (int): The time elapsed since the last frame.
            pipes (list): List of pipes in the game.
        """
        if self.state == BIRD_ALIVE:
            self.time_lived += dt
            self.move(dt)
            self.jump(pipes)
            self.draw()
            self.check_status(pipes)

    def get_inputs(self, pipes):
        """
        Generates inputs for the bird's neural network based on the closest pipe.

        Parameters:
            pipes (list): List of pipes in the game.

        Returns:
            list: Inputs for the neural network.
        """
        closest = DISPLAY_W * 2 
        bottom_y = 0  
        for p in pipes:
            if p.pipe_type == PIPE_UPPER and p.rect.right < closest and p.rect.right > self.rect.left:
                closest = p.rect.right
                bottom_y = p.rect.bottom

        horizontal_distance = closest - self.rect.centerx
        vertical_distance = (self.rect.centery) - (bottom_y + PIPE_GAP_SIZE / 2)

        inputs = [
            ((horizontal_distance / DISPLAY_W) * 0.99) + 0.01,
            ((( vertical_distance + Y_SHIFT) / NORMALIZER ) * 0.99 ) + 0.01
        ]

        return inputs

    @staticmethod
    def create_offspring(p1, p2, gameDisplay):
        """
        Creates a new bird offspring from two parent birds.

        Parameters:
            p1 (Bird): First parent bird.
            p2 (Bird): Second parent bird.
            gameDisplay (pygame.Surface): The game display surface.

        Returns:
            Bird: The new bird offspring.
        """
        new_bird = Bird(gameDisplay)
        new_bird.nnet.create_mixed_weights(p1.nnet, p2.nnet)
        return new_bird


class BirdCollection:
    """
    The BirdCollection class manages a collection of birds in the game.
    """

    def __init__(self, gameDisplay):
        """
        Initializes a BirdCollection.

        Parameters:
            gameDisplay (pygame.Surface): The game display surface.
        """
        self.gameDisplay = gameDisplay
        self.birds = []
        self.create_new_generation()

    def create_new_generation(self):
        """
        Creates a new generation of birds.
        """
        self.birds = []
        for i in range(0, GENERATION_SIZE):
            self.birds.append(Bird(self.gameDisplay))

    def update(self, dt, pipes):
        """
        Updates the position and state of all birds in the collection.

        Parameters:
            dt (int): The time elapsed since the last frame.
            pipes (list): List of pipes in the game.

        Returns:
            int: The number of birds alive.
        """
        num_alive = 0
        for b in self.birds:
            b.update(dt, pipes)
            if b.state == BIRD_ALIVE:
                num_alive += 1

        return num_alive

    def evolve_population(self):
        """
        Evolves the bird population based on their fitness.
        """
        for b in self.birds:
            b.fitness += b.time_lived * PIPE_SPEED

        self.birds.sort(key=lambda x: x.fitness, reverse=True)

        cut_off = int(len(self.birds) * MUTATION_CUT_OFF)
        good_birds = self.birds[0:cut_off]
        bad_birds = self.birds[cut_off:]
        num_bad_to_take = int(len(self.birds) * MUTATION_BAD_TO_KEEP)

        for b in bad_birds:
            b.nnet.modify_weights()

        new_birds = []

        idx_bad_to_take = np.random.choice(np.arange(len(bad_birds)), num_bad_to_take, replace=False)

        for index in idx_bad_to_take:
            new_birds.append(bad_birds[index])

        new_birds.extend(good_birds)

        children_needed = len(self.birds) - len(new_birds)

        while len(new_birds) < len(self.birds):
            idx_to_breed = np.random.choice(np.arange(len(good_birds)), 2, replace=False)
            if idx_to_breed[0] != idx_to_breed[1]:
                new_bird = Bird.create_offspring(good_birds[idx_to_breed[0]], good_birds[idx_to_breed[1]], self.gameDisplay)
                if random.random() < MUTATION_MODIFY_CHANCE_LIMIT:
                    new_bird.nnet.modify_weights()
                new_birds.append(new_bird)

        for b in new_birds:
            b.reset()

        self.birds = new_birds


# If the script is executed directly, run tests
if __name__ == "__main__":
    # Add any relevant tests for the classes here
    pass
