'''
Author: Daniel Eftekhari
Correspondences to daniel.eftekhari@mail.utoronto.ca

'''

import numpy as np
import matplotlib.pyplot as plt

def create_maze(load_maze, shape, prob_not_wall, prob_reward):
    if load_maze == False:
        maze = np.random.uniform(0, 1, (shape[0], shape[1]))
        maze = (maze > prob_not_wall)
        maze = maze.astype(np.float32)
        maze[:,0] = 1
        maze[int(shape[0]/2)-1:int(shape[0]/2)+2,0] = 0

        maze_area = np.copy(maze[:,2:])
        uniform = np.random.uniform(0, 1, (shape[0], shape[1]-2))
        rand = np.random.rand(shape[0], shape[1]-2)
        rand = (rand < prob_reward)
        maze_area[(maze_area != 1) * (rand == True)] = uniform[(maze_area != 1) * (rand == True)]
        maze[:,2:]= maze_area

        maze[maze == 0] = 2
        np.savetxt('maze1.txt', (maze), fmt="%f")
    else:
        maze = np.loadtxt('maze1.txt')
    save_maze(np.copy(maze), 1)

    return maze

def save_maze(maze, number):
    maze[maze == 2] = -2
    maze[maze == 1] = -3
    plt.imshow(maze, interpolation='nearest')
    plt.savefig('maze' + str(number) + '.png')

def initialize_prob(maze, shape):
    prob = np.zeros((shape[0], shape[1], 4))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if maze[i,j] > 1:
                if i >= 1 and i < (shape[0]-1) and j >= 1 and j < (shape[1]-1):
                    num_zeros = 1.0 * np.sum(maze[i-1,j] != 1) + 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i,j-1] != 1) + 1.0 * np.sum(maze[i,j+1] != 1)
                elif i == 0 and j == 0:
                    num_zeros = 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i,j+1] != 1)
                elif i == 0 and j > 0 and j < (shape[1]-1):
                    num_zeros = 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i,j-1] != 1) + 1.0 * np.sum(maze[i,j+1] != 1)
                elif i == 0 and j == (shape[1]-1):
                    num_zeros = 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i,j-1] != 1)
                elif i >= 1 and j == (shape[1]-1) and i < (shape[0]-1):
                    num_zeros = 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i-1,j] != 1) + 1.0 * np.sum(maze[i,j-1] != 1)
                elif i == (shape[0]-1) and j == (shape[1]-1):
                    num_zeros = 1.0 * np.sum(maze[i-1,j] != 1) + 1.0 * np.sum(maze[i,j-1] != 1)
                elif i == (shape[0]-1) and j == 0:
                    num_zeros = 1.0 * np.sum(maze[i,j+1] != 1) + 1.0 * np.sum(maze[i-1,j] != 1)
                elif i == (shape[0]-1) and j > 0 and j < (shape[1]-1):
                    num_zeros = 1.0 * np.sum(maze[i,j+1] != 1) + 1.0 * np.sum(maze[i,j-1] != 1) + 1.0 * np.sum(maze[i-1,j] != 1)
                elif i > 0 and i < (shape[0]-1) and j == 0:
                    num_zeros = 1.0 * np.sum(maze[i-1,j] != 1) + 1.0 * np.sum(maze[i+1,j] != 1) + 1.0 * np.sum(maze[i,j+1] != 1)

                try:
                    initial_prob = 1.0 / num_zeros
                except:
                    continue
                if j >= 1 and maze[i,j-1] != 1:
                    prob[i,j,0] = initial_prob
                if i >= 1 and maze[i-1,j] != 1:
                    prob[i,j,1] = initial_prob
                try:
                    if maze[i,j+1] != 1:
                        prob[i,j,2] = initial_prob
                except:
                    pass
                try:
                    if maze[i+1,j] != 1:
                        prob[i,j,3] = initial_prob
                except:
                    pass

    return prob

def update_maze(load_maze, change_values, maze, shape):
    if load_maze == False:
        if change_values == True:
            uniform = np.random.uniform(0, 1, (shape[0], shape[1]))
            maze[maze < 1] = uniform[maze < 1]
        np.savetxt('maze2.txt', (maze), fmt="%f")
    else:
        maze = np.loadtxt('maze2.txt')
    save_maze(np.copy(maze), 2)

    return maze

def show_position(maze, position, steps):
    maze[maze == 2] = -2
    maze[maze == 1] = -3
    if steps == 0:
        maze[position[0], position[1]] = -10
    else:
        maze[position[0], position[1]] = -5
    plt.imshow(maze, interpolation='nearest')
    plt.pause(0.0000001)
    plt.clf()

def save_policy(prob, iterations):
    optimal_policy = np.argmax(prob, axis=-1)
    np.savetxt('optimal_policy' + str(iterations) + '.txt', (optimal_policy), fmt="%d", newline='\r\n')

def save_path(path, iterations):
    np.savetxt('path' + str(iterations) + '.txt', (path), fmt="%d", newline='\r\n')
