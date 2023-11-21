import pygame
import random
from defs import *


class Pipe:
    """
    The Pipe class represents an individual pipe in the game.
    """

    def __init__(self, gameDisplay, x, y, pipe_type):
        """
        Initializes a pipe with its properties.

        Parameters:
            gameDisplay (pygame.Surface): The game display surface.
            x (int): The x-coordinate of the pipe.
            y (int): The y-coordinate of the pipe.
            pipe_type (int): The type of the pipe (PIPE_UPPER or PIPE_LOWER).
        """
        self.gameDisplay = gameDisplay
        self.state = PIPE_MOVING
        self.pipe_type = pipe_type
        self.img = pygame.image.load(PIPE_FILENAME)
        self.rect = self.img.get_rect()
        if pipe_type == PIPE_UPPER:
            y = y - self.rect.height
        self.set_position(x, y)

    def set_position(self, x, y):
        """
        Sets the position of the pipe.

        Parameters:
            x (int): The new x-coordinate.
            y (int): The new y-coordinate.
        """
        self.rect.left = x
        self.rect.top = y

    def move_position(self, dx, dy):
        """
        Moves the pipe by a specified amount.

        Parameters:
            dx (int): The amount to move along the x-axis.
            dy (int): The amount to move along the y-axis.
        """
        self.rect.centerx += dx
        self.rect.centery += dy

    def draw(self):
        """
        Draws the pipe on the game display.
        """
        self.gameDisplay.blit(self.img, self.rect)

    def check_status(self):
        """
        Checks the status of the pipe and updates its state if needed.
        """
        if self.rect.right < 0:
            self.state = PIPE_DONE

    def update(self, dt):
        """
        Updates the position and state of the pipe.

        Parameters:
            dt (int): The time elapsed since the last frame.
        """
        if self.state == PIPE_MOVING:
            self.move_position(-(PIPE_SPEED * dt), 0)
            self.draw()
            self.check_status()


class PipeCollection:
    """
    The PipeCollection class manages a collection of pipes in the game.
    """

    def __init__(self, gameDisplay):
        """
        Initializes a PipeCollection.

        Parameters:
            gameDisplay (pygame.Surface): The game display surface.
        """
        self.gameDisplay = gameDisplay
        self.pipes = []

    def add_new_pipe_pair(self, x):
        """
        Adds a pair of pipes (upper and lower) to the collection.

        Parameters:
            x (int): The x-coordinate where the pipes will be placed.
        """
        top_y = random.randint(PIPE_MIN, PIPE_MAX - PIPE_GAP_SIZE)
        bottom_y = top_y + PIPE_GAP_SIZE

        p1 = Pipe(self.gameDisplay, x, top_y, PIPE_UPPER)
        p2 = Pipe(self.gameDisplay, x, bottom_y, PIPE_LOWER)

        self.pipes.append(p1)
        self.pipes.append(p2)

    def create_new_set(self):
        """
        Creates a new set of pipes, clearing the existing collection.
        """
        self.pipes = []
        placed = PIPE_FIRST

        while placed < DISPLAY_W:
            self.add_new_pipe_pair(placed)
            placed += PIPE_ADD_GAP

    def update(self, dt):
        """
        Updates the position and state of all pipes in the collection.

        Parameters:
            dt (int): The time elapsed since the last frame.
        """
        rightmost = 0

        for p in self.pipes:
            p.update(dt)
            if p.pipe_type == PIPE_UPPER:
                if p.rect.left > rightmost:
                    rightmost = p.rect.left

        if rightmost < (DISPLAY_W - PIPE_ADD_GAP):
            self.add_new_pipe_pair(DISPLAY_W)

        self.pipes = [p for p in self.pipes if p.state == PIPE_MOVING]


# If the script is executed directly, run tests
if __name__ == "__main__":
    # Add any relevant tests for the classes here
    pass
