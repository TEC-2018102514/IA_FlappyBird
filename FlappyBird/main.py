# Importing necessary libraries and modules
import pygame
from defs import *
from pipe import PipeCollection
from bird import BirdCollection

def update_label(data, title, font, x, y, gameDisplay):
    """
    Updates a label on the game display.

    Parameters:
        data (any): The data to be displayed on the label.
        title (str): The title of the label.
        font (pygame.Font): The font used for the label.
        x (int): The x-coordinate of the label.
        y (int): The y-coordinate of the label.
        gameDisplay (pygame.Surface): The game display surface.

    Returns:
        int: The updated y-coordinate for the next label.
    """
    label = font.render('{} {}'.format(title, data), 1, DATA_FONT_COLOR)
    gameDisplay.blit(label, (x, y))
    return y

def update_data_labels(gameDisplay, dt, game_time, num_iterations, num_alive, font):
    """
    Updates various data labels on the game display.

    Parameters:
        gameDisplay (pygame.Surface): The game display surface.
        dt (int): The time elapsed since the last frame.
        game_time (int): The total time the game has been running.
        num_iterations (int): The number of game iterations.
        num_alive (int): The number of birds currently alive.
        font (pygame.Font): The font used for the labels.
    """
    y_pos = 10
    gap = 20
    x_pos = 10
    y_pos = update_label(round(1000/dt, 2), 'FPS', font, x_pos, y_pos + gap, gameDisplay)
    y_pos = update_label(round(game_time/1000, 2), 'Game time', font, x_pos, y_pos + gap, gameDisplay)
    y_pos = update_label(num_iterations, 'Iteration', font, x_pos, y_pos + gap, gameDisplay)
    y_pos = update_label(num_alive, 'Alive', font, x_pos, y_pos + gap, gameDisplay)

def run_game():
    """
    Main function to run the game.

    Initializes Pygame, sets up the game display, and runs the main game loop.
    """
    # Initializing Pygame
    pygame.init()
    
    # Creating the game display window
    gameDisplay = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption('Learn to fly')

    # Initializing game variables
    running = True
    bgImg = pygame.image.load(BG_FILENAME)
    pipes = PipeCollection(gameDisplay)
    pipes.create_new_set()
    birds = BirdCollection(gameDisplay)

    # Setting up font for data labels
    label_font = pygame.font.SysFont("monospace", DATA_FONT_SIZE)

    # Creating a clock to control the frame rate
    clock = pygame.time.Clock()
    dt = 0
    game_time = 0
    num_iterations = 1

    # Main game loop
    while running:
        dt = clock.tick(FPS)
        game_time += dt

        # Drawing the background image
        gameDisplay.blit(bgImg, (0, 0))

        # Handling events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = False

        # Updating pipes and birds
        pipes.update(dt)
        num_alive = birds.update(dt, pipes.pipes)

        # Checking if all birds are dead and triggering the next iteration
        if num_alive == 0:
            pipes.create_new_set()
            game_time = 0
            birds.evolve_population()
            num_iterations += 1

        # Updating and displaying data labels
        update_data_labels(gameDisplay, dt, game_time, num_iterations, num_alive, label_font)
        pygame.display.update()

# Running the game if the script is executed
if __name__ == "__main__":
    run_game()
