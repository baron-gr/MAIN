import pygame
import sys
import random
import time

pygame.init()
clock = pygame.time.Clock()

screen_W = 700
screen_H = 500
FPS = 60
screen = pygame.display.set_mode((screen_W, screen_H))
pygame.display.set_caption('Pong')

opponent = pygame.Rect(30, (screen_H/2) - 50, 15, 120)
player = pygame.Rect(screen_W - 40, (screen_H/2) - 50, 15, 120)
ball = pygame.Rect((screen_W/2) - 10, (screen_H/2) - 10, 20, 20)

bg_color = pygame.Color('grey12')
light_grey = (200, 200, 200)
white = (255, 255, 255)

ball_speed_x = -6
ball_speed_y = 6 * random.choice((1, -1))
player_speed = 0
opponent_speed = 6
lives = 3
seconds = 0
score = 0

font = pygame.font.Font('/Users/bgracias/MAIN/Python/Game-Dev/Pong-ML/Pong-Game/Consolas.ttf', 32)
game_over_font = pygame.font.Font('/Users/bgracias/MAIN/Python/Game-Dev/Pong-ML/Pong-Game/Consolas.ttf', 48)
game_over_text = game_over_font.render("GAME   OVER", True, white)

def ball_anim():
    global ball_speed_x, ball_speed_y, lives, lives_text, seconds, player_speed, score
    ball.x += ball_speed_x
    ball.y += ball_speed_y
    
    if ball.top <= 0 or ball.bottom >= screen_H:
        ball_speed_y *= -1
    if ball.left <= 0:
        score += 300
        ball_reset()
    if ball.colliderect(player) or ball.colliderect(opponent):
        if ball.colliderect(player):
            if abs(ball.right - player.left) < 10:
                ball_speed_x *= -1
                score += 200
            else:
                ball_speed_y *= -1
                score += 200
        elif ball.colliderect(opponent):
            if abs(ball.left - opponent.right) < 10:
                ball_speed_x *= -1
            else:
                ball_speed_y *= -1
    if ball.right >= screen_W:
        lives -= 1
        score -= 400
        if lives == 0:
            game_over()
        else:
            ball_reset()

def player_anim():
    global player_speed, score
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_H:
        player.bottom = screen_H
    if player.top < ball.y and player.bottom > ball.y:
        score += 1

def opponent_anim():
    if opponent.top < ball.y:
        opponent.top += opponent_speed
    if opponent.bottom > ball.y:
        opponent.bottom -= opponent_speed
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_H:
        opponent.bottom = screen_H

def ball_reset():
    global ball_speed_x, ball_speed_y
    ball.center = (screen_W/2, screen_H/2)
    opponent.centery = screen_H//2
    player.centery = screen_H//2
    ball_speed_x = -6
    ball_speed_y *= random.choice((1, -1))

def game_over():
    global lives, player_speed, seconds, record, timer, score
    screen.blit(game_over_text, (screen_W/2 - game_over_text.get_width()/2, screen_H/2 - game_over_text.get_height()/2))
    pygame.display.flip()
    waiting = True
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    ball_reset()
    lives = 3
    seconds = pygame.time.get_ticks()
    opponent.centery = screen_H/2
    player.centery = screen_H/2
    set_record(record, score)
    score = 0

def game_reset():
    global lives, player_speed, timer_text, seconds, record, timer, score
    pygame.time.wait(400)
    ball_reset()
    lives = 3
    seconds = pygame.time.get_ticks()
    player_speed = 0
    opponent.centery = screen_H/2
    player.centery = screen_H/2
    set_record(record, score)
    score = 0

def get_record():
    try:
        with open('/Users/bgracias/MAIN/Python/Game-Dev/Pong-ML/Pong-Game/record') as f:
            return f.readline()
    except FileNotFoundError:
        with open('/Users/bgracias/MAIN/Python/Game-Dev/Pong-ML/Pong-Game/record', 'w') as f:
            f.write('0')

def set_record(record, score):
    rec = max(int(record), int(score))
    with open('/Users/bgracias/MAIN/Python/Game-Dev/Pong-ML/Pong-Game/record', 'w') as f:
        f.write(str(rec))
    return rec

while True:
    record = get_record()
    lives_text = font.render("Lives: " + str(lives), True, white)
    timer = "{:.0f}".format((pygame.time.get_ticks() - seconds) / 1000)
    timer_text = font.render("Time: " + timer, True, white)
    score_text = font.render("Score: " + str(score), True, white)
    record_text = font.render("Record: " + str(record), True, white)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed += 7
            if event.key == pygame.K_UP:
                player_speed -= 7
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 7
            if event.key == pygame.K_UP:
                player_speed += 7
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_reset()
    
    ball_anim()
    player_anim()
    opponent_anim()
    
    screen.fill(bg_color)
    pygame.draw.rect(screen, light_grey, opponent)
    pygame.draw.rect(screen, light_grey, player)
    pygame.draw.ellipse(screen, light_grey, ball)
    pygame.draw.aaline(screen, light_grey, (screen_W/2,0), (screen_W/2,screen_H))
    screen.blit(lives_text, (10, 10))
    screen.blit(score_text, (10, screen_H - score_text.get_height() - 10))
    screen.blit(timer_text, (screen_W - timer_text.get_width() - 10, 10))
    screen.blit(record_text, (screen_W - record_text.get_width() - 10, screen_H - record_text.get_height() - 10))
    
    pygame.display.flip()
    clock.tick(FPS)