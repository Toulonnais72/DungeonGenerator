import pygame
from pygame.locals import *
import random, os
tilesize = 15

# Some variables and Video initiation
white = (255, 255, 255)
black = (0,0,0)
red = (180,0 ,0)
orange = (255, 159,0)
green = (50, 200, 50)

os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()
infoObject = pygame.display.Info()
X = infoObject.current_w
Y = infoObject.current_h
fontsize = int(Y/40)
left_col = int(X/5)
right_col = int(X/1.8)
display_surface = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Dungeon Generator')


class spritesheet:

    def __init__(self, filename, cols, rows):
        self.sheet = pygame.image.load(filename).convert_alpha()
        self.rect = self.sheet.get_rect()
        self.sheet = pygame.transform.scale(self.sheet, (int(self.rect.height * tilesize / 150), int(self.rect.width * tilesize / 150)))
        self.rect = self.sheet.get_rect()
        self.cols = cols
        self.rows = rows
        self.totalCellCount = cols * rows

        w = self.cellWidth = int(self.rect.width / cols)
        h = self.cellHeight = int(self.rect.height / rows)
        hw, hh = self.cellCenter = (int(w / 2), int(h / 2))

        self.cells = list([(index % cols * w, int(index / cols) * h, w, h) for index in range(self.totalCellCount)])
        self.handle = list([
            (0, 0), (-hw, 0), (-w, 0),
            (0, -hh), (-hw, -hh), (-w, -hh),
            (0, -h), (-hw, -h), (-w, -h), ])

    def draw(self, surface, cellIndex, x, y, handle=0):
        surface.blit(self.sheet, (x + self.handle[handle][0], y + self.handle[handle][1]), self.cells[cellIndex])


tileset = spritesheet("./images/tileset/tileset2.png", 16, 16)
greytiles = spritesheet("./images/tileset/greytiles.png", 4,4)
greentiles = spritesheet("./images/tileset/greentiles.png", 4, 4)



