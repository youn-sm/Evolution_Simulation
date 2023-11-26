classes.py

import random
import numpy as np
import pygame

WIDTH, HEIGHT = 1000, 1000
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

FPS = 60
NUM_PREDATORS = 30
NUM_PREY = 50
PREDATOR_SIZE = 7
PREY_SIZE = 7
SIMULATION_DURATION = -1

# 상호작용에 관여하는 상수들
PREDATOR_SPEED = 15
PREY_SPEED = 10
MAX_TIME_WITHOUT_FOOD = 6
predator_reproduction_delay = 1
prey_reproduction_delay = 1
CHASING_RANGE = 90
RUNNING_AWAY_RANGE = 50


class Predator:
    # 초깃값: 랜덤 생성 위치, 먹이 없이 생존 가능 시간 = 0, 가장 최근 복제 후 지난 시간 = 0, 시뮬레이션 시작시 이동방향 랜덤, 피식자 쫓기 = None
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.time_without_food = 0
        self.last_reproduction_time = 0
        self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.direction /= np.linalg.norm(self.direction)
        self.chasing_prey = None
    # 포식자 움직임
    def move(self):
        self.x += self.direction[0] * PREDATOR_SPEED
        self.y += self.direction[1] * PREDATOR_SPEED

        # 화면 경계와 닿았는지 확인
        if self.x < 0 or self.x >= WIDTH:
            # 방향 바꾸기
            self.direction[0] *= -1

        if self.y < 0 or self.y >= HEIGHT:
            self.direction[1] *= -1

        # 이동 방향 랜덤으로 바꾸기
        if random.random() < 0.1:
            self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            # 방향 설정에 필요한 단위 벡터 구하기
            self.direction /= np.linalg.norm(self.direction)

    # 포식자가 피식자 먹으면 100% 확률로 복제
    def can_reproduce(self, current_time):
        return current_time - self.last_reproduction_time >= predator_reproduction_delay

    # 포식자 그리기
    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), PREDATOR_SIZE)

# 피식자 클래스
class Prey:
    # 초깃값: 랜덤 생성 위치, 시뮬레이션 시작시 이동방향 랜덤, 랜덤 시간 안에 복제, 가장 최근 복제 후 지난 시간 = 0, 현재 시간 = 0
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.direction /= np.linalg.norm(self.direction)
        self.reproduction_delay = prey_reproduction_delay
        self.last_reproduction_time = 0
        self.speed = PREY_SPEED #PREY_SPEED는 속도의 기본 초기값

    # 피식자 움직이기
    def move(self, predators):
        # 시야 범위를 나타내는 각도

        # 범위 안에 포식자가 있는지 확인
        for predator in predators:
            prey_to_predator_vector = np.array([predator.x - self.x, predator.y - self.y])
            distance = np.linalg.norm(prey_to_predator_vector)

            # 시야 범위 안에 있는지 확인
            if distance < RUNNING_AWAY_RANGE:

                # 포식자로부터 도망치기
                opposite_direction = -prey_to_predator_vector
                self.direction = opposite_direction
                self.direction /= np.linalg.norm(self.direction)
                self.x += self.direction[0] * self.speed
                self.y += self.direction[1] * self.speed
                return

        # 시야 안에 포식자가 없다면 평소처럼 움직이기
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed

        # 화면 경계와 닿았는지 확인
        if self.x < 0 or self.x >= WIDTH:
            # 방향 바꾸기
            self.direction[0] *= -1

        if self.y < 0 or self.y >= HEIGHT:
            self.direction[1] *= -1

        # 이동방향 랜덤으로 바꾸기
        if random.random() < 0.1:
            self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            # 방향 설정에 필요한 단위 벡터 구하기
            self.direction /= np.linalg.norm(self.direction)

    # 피식자 그리기
    def draw(self,screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), PREY_SIZE)

    # 피식자 복제하기
    def can_reproduce(self, current_time):
        if current_time - self.last_reproduction_time >= self.reproduction_delay:
            return True
        return False
