import torch
import random
import numpy as np
from collections import deque
from snake_AI import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_Qnet, Qtrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = .001   # learning rate


class Agent() :
    def __init__(self) -> None:
        self.n_games = 0    # I have to comment that ?
        self.epsilon = 0    # for the random part
        self.gamma = 0.9      # discount rate, need to be under 1
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3)    # 11 and 3 are important but we can play with the 256
        self.trainner = Qtrainer(LR, self.model, self.gamma)    
        
    
    def get_state(self, game:SnakeGameAI)  :
        head = game.snake[0]

        point_l = Point(head.x + BLOCK_SIZE, head.y)
        point_r = Point(head.x - BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y + BLOCK_SIZE)
        point_d = Point(head.x, head.y - BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straigth
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger rigth
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        
        return np.array(state, dtype=int)
    
    def  remember(self, state, action:list, reward:int, next_stat, done) -> None :
        # pop the left item if len memory go over MAX_MEMORY
        self.memory.append((state, action, reward, next_stat, done))

    def train_long_memory(self) -> None :
        if len(self.memory) > BATCH_SIZE :
            # make a random sample of data from the memory
            mini_sample = random.sample(self.memory, BATCH_SIZE)    # -> list of tuple
        else : 
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainner.training_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action:list, reward:int, next_state, done) -> None :
        self.trainner.training_step(state, action, reward, next_state, done)

    def get_action(self, state) -> list :
        # it make random move during the exploration phase
        self.epsilon = 80 - self.n_games
        final_move= [0, 0, 0]
        if random.randint(0, 200) < self.epsilon :
            move = random.randint(0, 2) 
            final_move[move] = 1
        else :
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def train() :
    plot_score = list()
    plot_mean_score = list()
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    running = True
    while  running :
        state_old = agent.get_state(game)   # old state of the game

        final_move = agent.get_action(state_old)    # get the move

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # traning short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done : 
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record :
                record = score
                agent.model.save()
                
        
            print(f"nbr Game : {agent.n_games} | Score : {score} | record : {record}")

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

    

if __name__ == "__main__" :
    train()