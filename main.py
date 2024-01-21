import time
import pygame
import torch as th
from pygame.locals import HWSURFACE, DOUBLEBUF
from ants.ants import Ants
from debug.fps_counter import FpsCounter


th.set_grad_enabled(False)
device = "cuda:0" if th.cuda.is_available() else "cpu"
device = "cuda:0"
resolution = 1
ants_count = 1000000

fps_counter = FpsCounter()

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1600 - 1600 % resolution, 900 - 900 % resolution
screen_size = th.tensor((WIDTH, HEIGHT), dtype=th.float32)
screen = pygame.display.set_mode((WIDTH, HEIGHT), HWSURFACE | DOUBLEBUF)

# Ants
ants = Ants(screen_size, resolution, ants_count, dtype=th.float32, device=device)

# Main loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Key down event
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Fill the screen with white
    screen.fill((255, 255, 255))

    # Update ants
    ants.turn()
    ants.move()
    ants.diffuse()

    # Render ants
    texture = ants.render()
    screen.blit(texture, (0, 0))
    pygame.display.flip()
    fps_counter.update()
    pygame.display.set_caption(f"{fps_counter.get()} fps")
    time.sleep(0.01)

# Quit Pygame
pygame.quit()
