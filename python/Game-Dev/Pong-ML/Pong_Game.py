import pygame
import sys
import random

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

ball_speed_x = 7 * random.choice((1, -1))
ball_speed_y = 7 * random.choice((1, -1))
player_speed = 0
opponent_speed = 7
lives = 3

font = pygame.font.SysFont(None, 48)
game_over_text = font.render("GAME  OVER", True, white)

def ball_anim():
    global ball_speed_x, ball_speed_y, lives, lives_text
    ball.x += ball_speed_x
    ball.y += ball_speed_y
    
    if ball.top <= 0 or ball.bottom >= screen_H:
        ball_speed_y *= -1
    if ball.left <= 0:
        ball_reset()
    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1
        
    if ball.right >= screen_W:
        lives -= 1
        lives_text = font.render("Lives: " + str(lives), True, white)
        if lives == 0:
            screen.blit(game_over_text, (screen_W/2 - game_over_text.get_width()/2, screen_H/2 - game_over_text.get_height()/2))
            clock.tick(200)
            pygame.display.flip()
            pygame.time.wait(3000)
            game_reset()
        else:
            ball_reset()

def player_anim():
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_H:
        player.bottom = screen_H

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
    ball_speed_y *= random.choice((1, -1))
    ball_speed_x *= random.choice((1, -1))

def game_reset():
    global lives
    lives = 3
    ball_reset()

while True:
    lives_text = font.render("Lives: " + str(lives), True, white)

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
    
    ball_anim()
    player_anim()
    opponent_anim()
    
    screen.fill(bg_color)
    pygame.draw.rect(screen, light_grey, opponent)
    pygame.draw.rect(screen, light_grey, player)
    pygame.draw.ellipse(screen, light_grey, ball)
    pygame.draw.aaline(screen, light_grey, (screen_W/2,0), (screen_W/2,screen_H))
    screen.blit(lives_text, (10, 10))
    
    pygame.display.flip()
    clock.tick(FPS)