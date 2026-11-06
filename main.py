import pygame, math
import numpy as np
from random import uniform


SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 800
SIMULATION_STEPS = 5
SIMULATION_SPEED = 3

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
running = True
dt = 0

track = pygame.image.load("track.png").convert_alpha()
track = pygame.transform.scale(track, (SCREEN_WIDTH, SCREEN_HEIGHT))



class Car:
    def __init__(self):
        self.pos = pygame.Vector2(700, 558)
        self.rect = pygame.Rect(0, 0, 6, 10)
        self.rect.center = self.pos

        self.angle = 78.66
        self.surface = pygame.Surface((6, 10), pygame.SRCALPHA)
        self.surface.fill("#800000")
        self.speed = 0.0
        self.acceleration = 300
        self.friction = 300.0
        self.max_speed = 600.0
        self.steering = 0.0

        self.reward = 0
        self.living = True
        self.lifetime = 0
        self.network = NeuralNetwork()         ##call neural net per car

        self.output = np.zeros(2)


    def carMovement(self):
        global dt
        throttle = max(0.2, self.output[0])
        self.steering = self.output[1]

        self.angle += 180 * dt * self.steering * (abs(self.speed) / self.max_speed)

        
        if abs(throttle) > 0.01:
            self.speed += self.acceleration * dt * throttle
        else:
            if self.speed > 0:
                self.speed -= self.friction * dt
                self.speed = max(0, self.speed)
            elif self.speed < 0:
                self.speed += self.friction * dt
                self.speed = min(0, self.speed)
        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))

        angle_rad = math.radians(self.angle)

        direction = pygame.Vector2(
            -math.sin(angle_rad),
            -math.cos(angle_rad)
        )


        self.pos += direction * self.speed * dt

        self.rect.center = (round(self.pos.x), round(self.pos.y))
        self.rect.clamp_ip(screen.get_rect())

    def getInputs(self):
            return [
                self.sensor(-90, 100),
                self.sensor(-60, 120 ),
                self.sensor(-30, 150),
                self.sensor(0, 200),
                self.sensor(30, 150),
                self.sensor(60, 120),
                self.sensor(90, 100),
                self.speed / self.max_speed,
                self.steering
                ] 
    
    def sensor(self, angle, length):
        angle_rad = math.radians(self.angle + angle)

        dir_x = -math.sin(angle_rad)
        dir_y = -math.cos(angle_rad)

        for i in range(length): #perhaps adapt sensor the check for transition of track to grass? 
            check_x = self.pos.x + dir_x * i
            check_y = self.pos.y + dir_y * i

            if not (0 <= check_x < SCREEN_WIDTH and 0 <= check_y < SCREEN_HEIGHT):
                return i / length

            r, g, b, a = track.get_at((int(check_x), int(check_y)))


            if a == 0: 
                return i / length
        return 1.0


    
    def rewardProgramm(self):
        location_color = track.get_at((int(self.pos.x), int(self.pos.y)))
        r, g, b, a = location_color

        if abs(self.speed) < 1:
            self.reward -= 0.1

        if a > 253 and r < 10 and g < 10 and b < 10: #on track
            self.reward += 0.1
        elif a == 0: #off track
            #self.reward -= 0.2
            self.living = False 
        elif r == g and g == b: #finish line bby
            self.reward += 0
        #else: ##############PERHAPS ADD MILESTONES IN YELLOW ACROSSE THE TRACK?
            #print(f"UNKNOWN: {location_color}")

    def update(self):
        global dt

        self.lifetime += dt

        if self.lifetime > 30:
            self.living = False
            return

        inputs = self.getInputs()

        self.output = self.network.neuralNet(inputs)

        self.carMovement()

        self.rewardProgramm()

    def resetCar(self):
        self.speed = 0
        self.pos = pygame.Vector2(700, 558)
        self.angle = 78.66
        self.living = True
        self.reward = 0
        self.steering = 0
        self.lifetime = 0
        self.output = np.zeros(2)
        self.rect.center = self.pos






####  Neural Net  ####

#prev steering
class NeuralNetwork:
    def __init__(self):
        self.input_n = 9
        self.medial_n = 16
        self.output_n = 2

        self.synone = np.random.uniform(-1.0, 1.0, (self.input_n, self.medial_n))
        self.syntwo = np.random.uniform(-1.0, 1.0, (self.medial_n, self.output_n))
        self.medialin = np.zeros(self.medial_n)
        self.medialout = np.zeros(self.medial_n)
        self.output = np.zeros(self.output_n)


    def neuralNet(self, inputs):
        #net itself
        for i in range(self.medial_n):
            self.medialin[i] = 0
            for j in range(self.input_n):
                self.medialin[i] = self.medialin[i] + self.synone[j, i] * inputs[j]
            self.medialout[i] = math.tanh(self.medialin[i])

        for i in range(self.output_n):
            self.output[i] = 0
            for j in range(self.medial_n):
                self.output[i] = self.output[i] + self.syntwo[j, i] * self.medialout[j]
            self.output[i] = math.tanh(self.output[i])

        return self.output





def geneticElitism(cars):
    if all(not car.living for car in cars):
        elite_count = 10

        cars.sort(key=lambda car: car.reward, reverse = True)

        elite = cars[:elite_count]
        elite_networks = [cars[i].network for i in range(elite_count)]

        new_cars = []

        
        best_car = Car()
        best_car.network.synone = elite_networks[0].synone.copy()
        best_car.network.syntwo = elite_networks[0].syntwo.copy()
        new_cars.append(best_car)


        while len(new_cars) < len(cars):
            parent = elite_networks[np.random.randint(elite_count)]

            child = Car()

            child.network.synone = parent.synone.copy()
            child.network.syntwo = parent.syntwo.copy()

            Mutate(child.network)

            new_cars.append(child)

    return new_cars




def Mutate(network, mutation_strength = 0.1):
    network.synone += np.random.normal(0, mutation_strength, network.synone.shape)
    network.syntwo += np.random.normal(0, mutation_strength, network.syntwo.shape)

    return network


    
generation = 0

#main loop
cars = [Car() for _ in range(100)]
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    dt = clock.tick(60) / 1000 # limits FPS to 60
    dt *= SIMULATION_SPEED / SIMULATION_STEPS



    screen.fill("#3B7A37")
    screen.blit(track, (0, 0))

    for _ in range(SIMULATION_STEPS):
        for car in cars:
            if car.living:
                car.update()

    for car in cars:
        rotated_surface = pygame.transform.rotate(car.surface, car.angle)
        rotated_rect = rotated_surface.get_rect(center=car.rect.center)
        screen.blit(rotated_surface, rotated_rect)


    if all(not car.living for car in cars):
        best_reward = max(car.reward for car in cars)

        generation += 1 
        cars = geneticElitism(cars)

        print(f"generation: {generation}")
        print(f"highest reward: {best_reward}")
    


    #drawing

    pygame.display.flip() 




pygame.quit()

