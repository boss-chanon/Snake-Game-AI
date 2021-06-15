import math
import os
import pygame
import random
import time

FOLDER = os.path.dirname(__file__)
IMAGE_FOLDER = os.path.join(FOLDER, "image")

HEIGHT = 20
WIDTH = 20
BOX_SIZE = 20
PIXEL_H = BOX_SIZE * HEIGHT
PIXEL_W = BOX_SIZE * WIDTH

FPS = 10

GAME_TITLE = "Snake Game"

BG_IMAGE = os.path.join(IMAGE_FOLDER, "bg.png")
HEAD_IMAGE = os.path.join(IMAGE_FOLDER, "head.png")
BODY_IMAGE = os.path.join(IMAGE_FOLDER, "body.png")
FOOD_IMAGE = os.path.join(IMAGE_FOLDER, "food.png")

class ENV:
    def __init__(self, AI_active = False, limit_speed = True, render = True):
        self.reward = 0
        self.state_space = 12
        self.action_space = 4
        self.running =True
        self.render = render
        self.AI_active = AI_active
        self.limit_speed = limit_speed

        random.seed(time.time())

        if self.render:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption(GAME_TITLE)

            self.window = pygame.display.set_mode((PIXEL_W, PIXEL_H))

            self.font = pygame.font.SysFont("Angsana New", 36)
            self.bg_image = pygame.transform.scale(pygame.image.load(BG_IMAGE), (PIXEL_W, PIXEL_H))
            self.head_image = pygame.transform.scale(pygame.image.load(HEAD_IMAGE), (BOX_SIZE, BOX_SIZE))
            self.body_image = pygame.transform.scale(pygame.image.load(BODY_IMAGE), (BOX_SIZE, BOX_SIZE))
            self.food_image = pygame.transform.scale(pygame.image.load(FOOD_IMAGE), (BOX_SIZE, BOX_SIZE))

            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.done = False

        self.snake_len = 0
        self.point = 0

        self.direction = "stop"

        self.head_x = random.randint(0, WIDTH - 1)
        self.head_y = random.randint(0, HEIGHT - 1)
        self.body_x = []
        self.body_y = []
        self.move_food()

        self.dist = math.sqrt(((self.head_x - self.food_x) ** 2) + ((self.head_y - self.food_y) ** 2))

        state = self.get_state()
        
        return state

    def move_food(self):
        self.food_x = random.randint(0, WIDTH - 1)
        self.food_y = random.randint(0, HEIGHT - 1)

        if not self.food_check():
            self.move_food()

    def food_check(self):
        for i in range(len(self.body_x)):
            if self.food_x == self.body_x[i] and self.food_y == self.body_y[i]:
                return False
        
        return True

    def move(self):
        if self.direction == "stop":
            x_move = 0
            y_move = 0
        if self.direction == "up":
            x_move = 0
            y_move = -1
        if self.direction == "down":
            x_move = 0
            y_move = 1
        if self.direction == "left":
            x_move = -1
            y_move = 0
        if self.direction == "right":
            x_move = 1
            y_move = 0
        self.head_x += x_move
        self.head_y += y_move

    def go_up(self):
        if self.direction != "down":
            self.direction = "up"

    def go_down(self):
        if self.direction != "up":
            self.direction = "down"

    def go_right(self):
        if self.direction != "left":
            self.direction = "right"

    def go_left(self):
        if self.direction != "right":
            self.direction = "left"

    def move_body(self):
        body_x = self.head_x
        body_y = self.head_y

        self.body_x.append(body_x)
        self.body_y.append(body_y)

        if len(self.body_x) > self.snake_len:
            self.body_x.pop(0)
        if len(self.body_y) > self.snake_len:
            self.body_y.pop(0)

    def measure_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt(((self.head_x - self.food_x) ** 2) + ((self.head_y - self.food_y) ** 2))

    def eat_check(self):
        if (self.head_x == self.food_x) and (self.head_y == self.food_y):
            return True
        return False

    def death_check(self):
        for n in range(len(self.body_x[3:])):
            if (self.head_x == self.body_x[n]) and (self.head_y == self.body_y[n]):
                return True
        if (self.head_x == -1) or (self.head_x == WIDTH) or (self.head_y == -1) or (self.head_y == HEIGHT):
            return True
        return False

    def get_state(self):
        if self.head_y - 1 == -1:
            wall_up, wall_down = 1, 0
        elif self.head_y + 1 == HEIGHT:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0

        if self.head_x - 1 == -1:
            wall_left, wall_right = 1, 0
        elif self.head_x + 1 == WIDTH:
            wall_left, wall_right = 0, 1
        else:
            wall_left, wall_right = 0, 0

        body_up = 0
        body_right = 0
        body_down = 0
        body_left = 0
        for y in self.body_y:
            if self.head_y - 1 == y:
                body_up = 1
            if self.head_y + 1 == y:
                body_down = 1
        for x in self.body_x:
            if self.head_x - 1 == x:
                body_left = 1
            if self.head_x + 1 == x:
                body_right = 1

        state = [int(self.food_x > self.head_x), int(self.food_y > self.head_y), int(self.head_x > self.food_x), int(self.head_y > self.food_y), \
                int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                int(self.direction == 'up'), int(self.direction == 'right'), int(self.direction == 'down'), int(self.direction == 'left')]

        return state

    def game_run(self):
        if self.limit_speed:
            self.clock.tick(FPS)

        self.move()
        self.move_body()

        self.measure_distance()

        if self.dist < self.prev_dist:
            self.reward = 1
        if self.dist > self.prev_dist:
            self.reward = -1

        if self.eat_check():
            self.snake_len += 1
            self.point += 1
            self.reward = 10
            self.move_food()

        if self.death_check():
            self.done = True
            self.reward = -100
            if not self.AI_active:
                self.reset()

        state = self.get_state()
        
        if self.render:
            self.text = self.font.render("Point: " + str(self.point), True, (255, 255, 255))

        return state, self.reward, self.done

    def show(self):
        if self.render:
            self.quit()

            self.window.blit(self.bg_image, (0, 0))
            self.window.blit(self.food_image, (self.food_x * BOX_SIZE, self.food_y * BOX_SIZE))
            self.window.blit(self.head_image, (self.head_x * BOX_SIZE, self.head_y * BOX_SIZE))
            for n in range(len(self.body_x)):        
                self.window.blit(self.body_image, (self.body_x[n] * BOX_SIZE, self.body_y[n] * BOX_SIZE))
            self.window.blit(self.text, (0, 0))

            pygame.display.flip()

    def quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

if __name__ == "__main__":
    game = ENV()

    while game.running:

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            game.go_up()
        if pressed[pygame.K_DOWN]:
            game.go_down()
        if pressed[pygame.K_LEFT]:
            game.go_left()
        if pressed[pygame.K_RIGHT]:
            game.go_right()

        game.game_run()
        game.show()
    
    pygame.quit()