import pygame
from copy import deepcopy
from random import choice

W, H = 10, 20
TILE = 40
GAME_RES = W * TILE, H * TILE
RES = 750, 830
FPS = 60

pygame.init()
sc = pygame.display.set_mode(RES)
game_sc = pygame.Surface(GAME_RES)
clock = pygame.time.Clock()

grid = [pygame.Rect(x * TILE, y * TILE, TILE, TILE) for x in range(W) for y in range(H)]

figures_pos = [[(-1, 0),(-2, 0),(0, 0),(1, 0)],
               [(0, -1),(-1, -1),(-1, 0),(0, 0)],
               [(-1, 0),(-1, 1),(0, 0),(0, -1)],
               [(0, 0),(-1, 0),(0, 1),(-1, -1)],
               [(0, 0),(0, -1),(0, 1),(-1, -1)],
               [(0, 0),(0, -1),(0, 1),(1, -1)],
               [(0, 0),(0, -1),(0, 1),(-1, 0)]]

figures = [[pygame.Rect(x + W // 2, y + 1, 1, 1) for x, y in fig_pos] for fig_pos in figures_pos]
figure_rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
field = [[0 for i in range(W)] for j in range(H)]

anim_count, anim_speed, anim_limit = 0, 6, 2000

bg = pygame.image.load('/Users/bgracias/MAIN/Python/Games/Tetris/space_bg.jpg').convert()
game_bg = pygame.image.load('/Users/bgracias/MAIN/Python/Games/Tetris/stars_bg.jpg').convert()

main_font = pygame.font.Font('/Users/bgracias/MAIN/Python/Games/Tetris/Tetris.ttf', 70)
font = pygame.font.Font('/Users/bgracias/MAIN/Python/Games/Tetris/Tetris.ttf', 55)

title_tetris = main_font.render('TETRIS', True, pygame.Color('darkorange'))
title_record = font.render('Record', True, pygame.Color('purple'))
title_score = font.render('Score', True, pygame.Color('green'))

# get_color = lambda : (randrange(30, 256), randrange(30, 256), randrange(30, 256))

get_color = lambda: choice([
    (255, 0, 0),     # red
    (255, 128, 0),   # orange
    (255, 255, 0),   # yellow
    (0, 255, 0),     # green
    (0, 0, 255),     # blue
    (75, 0, 130),    # indigo
    (238, 130, 238), # violet
])

figure, next_figure = deepcopy(choice(figures)), deepcopy(choice(figures))
color, next_color = get_color(), get_color()

score, lines = 0, 0
scores = {0: 0, 1: 1000, 2: 5000, 3: 12000, 4: 30000}

def check_borders():
    if figure[i].x < 0 or figure[i].x > W - 1:
        return False
    elif figure[i].y > H - 1 or field[figure[i].y][figure[i].x]:
        return False
    return True

def get_record():
    try:
        with open('/Users/bgracias/MAIN/Python/Games/Tetris/record') as f:
            return f.readline()
    except FileNotFoundError:
        with open('/Users/bgracias/MAIN/Python/Games/Tetris/record', 'w') as f:
            f.write('0')

def set_record(record, score):
    rec = max(int(record), score)
    with open('/Users/bgracias/MAIN/Python/Games/Tetris/record', 'w') as f:
        f.write(str(rec))

while True:
    record = get_record()
    dx, rotate = 0, False
    save_piece = False
    sc.blit(bg, (0, 0))
    sc.blit(game_sc, (20, 20))
    game_sc.blit(game_bg, (0, 0))
    
    # delay for full lines
    for i in range(lines):
        pygame.time.wait(200)
    
    # control
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                dx = -1
            elif event.key == pygame.K_RIGHT:
                dx = 1
            elif event.key == pygame.K_DOWN:
                anim_limit = 400
            elif event.key == pygame.K_UP:
                rotate = True
            elif event.key == pygame.K_SPACE:
                save_piece = True

    # move x
    figure_old = deepcopy(figure)
    for i in range(4):
        figure[i].x += dx
        if not check_borders():
            figure = deepcopy(figure_old)
            break
    
    # move y
    anim_count += anim_speed
    if anim_count > anim_limit:
        anim_count = 0
        figure_old = deepcopy(figure)
        for i in range(4):
            figure[i].y += 1
            if not check_borders():
                for i in range(4):
                    field[figure_old[i].y][figure_old[i].x] = color
                figure, color = next_figure, next_color
                next_figure, next_color = deepcopy(choice(figures)), get_color()
                anim_limit = 2000
                break
            
    # rotate
    centre = figure[0]
    figure_old = deepcopy(figure)
    if rotate:
        for i in range(4):
            x = figure[i].y - centre.y
            y = figure[i].x - centre.x
            figure[i].x = centre.x - x
            figure[i].y = centre.y + y
            if not check_borders():
                figure = deepcopy(figure_old)
                break
    
    # save piece
    
    
    # check lines
    line, lines = H - 1, 0
    for row in range(H - 1, -1, -1):
        count = 0
        for i in range(W):
            if field[row][i]:
                count += 1
            field[line][i] = field[row][i]
        if count < W:
            line -= 1
        else:
            anim_speed += 0.1
            lines += 1
    
    # compute score
    score += scores[lines]
    
    # draw grid
    [pygame.draw.rect(game_sc, (40, 40, 40), i_rect, 1) for i_rect in grid]
    
    # draw figure
    for i in range(4):
        figure_rect.x = figure[i].x * TILE
        figure_rect.y = figure[i].y * TILE
        pygame.draw.rect(game_sc, color, figure_rect)
    
    # draw field
    for y, raw in enumerate(field):
        for x, col in enumerate(raw):
            if col:
                figure_rect.x, figure_rect.y = x * TILE, y * TILE
                pygame.draw.rect(game_sc, col, figure_rect)
    
    # draw next figure
    for i in range(4):
        figure_rect.x = next_figure[i].x * TILE + 390
        figure_rect.y = next_figure[i].y * TILE + 190
        pygame.draw.rect(sc, next_color, figure_rect)
    
    # draw titles
    sc.blit(title_tetris, (440, 40))
    sc.blit(title_record, (490, 400))
    sc.blit(title_score, (505, 600))
    sc.blit(font.render(record, True, pygame.Color('gold')), (530, 480))
    sc.blit(font.render(str(score), True, pygame.Color('white')), (535, 680))
    
    # game over
    for i in range(W):
        if field[0][i]:
            set_record(record, score)
            field = [[0 for i in range(W)] for i in range(H)]
            anim_count, anim_speed, anim_limit = 0, 6, 2000
            score = 0
            for i_rect in grid:
                pygame.draw.rect(game_sc, get_color(), i_rect)
                sc.blit(game_sc, (20, 20))
                pygame.display.flip()
                clock.tick(200)
    
    pygame.display.flip()
    clock.tick()