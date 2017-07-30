'''
Author: Daniel Eftekhari
Correspondences to daniel.eftekhari@mail.utoronto.ca

A novel reinforcement learning algorithm (see README for details)

The program saves the initial and final mazes (.png and .txt).
The program saves the path that was taken just before rewards are changed,
or equivalently halfway through the number of iterations, and at program termination.
The program saves the most probable moves for each position in the maze just before rewards are changed,
or equivalently halfway through the number of iterations, and at program termination.
'''

import numpy as np
import time
import matplotlib.pyplot as plt

def create_maze(load_maze, x_row, y_row, prob_not_wall, prob_reward):
    if load_maze == False:
        maze = np.random.uniform(0, 1, (x_row, y_row))
        maze = (maze > prob_not_wall)
        maze = maze.astype(np.float32)
        maze[:,0] = 1
        maze[int(x_row/2)-1:int(x_row/2)+2,0] = 0

        for i in range(maze.shape[0]):
            for j in range(2, maze.shape[1]):
                if maze[i,j] != 1:
                    random_number = np.random.rand(1)
                    if random_number < prob_reward:
                        maze[i,j] = np.random.uniform(0, 1)

        maze[maze == 0] = 2
        np.savetxt('maze1.txt', (maze), fmt="%f")
    else:
        maze = np.loadtxt('maze1.txt')

    save_maze(np.copy(maze), 1)

    return maze

def save_maze(maze, number):
    maze = maze
    maze[maze == 2] = -2
    maze[maze == 1] = -3
    plt.imshow(maze, interpolation='nearest')
    plt.savefig('maze' + str(number) + '.png')

def initialize_prob(maze):
    shape = np.shape(maze)
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

def update_maze(load_maze, change_values, maze):
    if load_maze == False:
        if change_values == True:
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    if maze[i,j] < 1:
                        maze[i,j] = np.random.uniform(0, 1)
        np.savetxt('maze2.txt', (maze), fmt="%f")
    else:
        maze = np.loadtxt('maze2.txt')
    save_maze(np.copy(maze), 2)

    return maze

def forward_pass(plot_maze, maze, prob):
    path = np.array([[maze.shape[0]/2,0]], dtype=np.uint16)
    steps = 0
    while maze[path[-1,0], path[-1,1]] > 1:
        # show agent position in maze
        if plot_maze == True:
            show_position(np.copy(maze), path[-1], steps)

        counter = 0
        random_number = np.random.rand(1)
        if random_number < (prob[path[steps,0],path[steps,1],0] + counter) and random_number >= counter:
            new_step = np.array([[path[steps,0],path[steps,1]-1]])
            path = np.concatenate((path, new_step))
            steps += 1
            continue
        else:
            counter += prob[path[steps,0],path[steps,1],0]
        if random_number < (prob[path[steps,0],path[steps,1],1] + counter) and random_number >= counter:
            new_step = np.array([[path[steps,0]-1,path[steps,1]]])
            path = np.concatenate((path, new_step))
            steps += 1
            continue
        else:
            counter += prob[path[steps,0],path[steps,1],1]
        try:
            if random_number < (prob[path[steps,0],path[steps,1],2] + counter) and random_number >= counter:
                new_step = np.array([[path[steps,0],path[steps,1]+1]])
                path = np.concatenate((path, new_step))
                steps += 1
                continue
            else:
                counter += prob[path[steps,0],path[steps,1],2]
        except:
            pass
        try:
            if random_number < (prob[path[steps,0],path[steps,1],3] + counter) and random_number >= counter:
                new_step = np.array([[path[steps,0]+1,path[steps,1]]])
                path = np.concatenate((path, new_step))
                steps += 1
                continue
            else:
                counter += prob[path[steps,0],path[steps,1],3]
        except:
            pass

    reward = maze[path[steps,0], path[steps,1]]

    return path, reward, steps

def show_position(maze, position, steps):
    maze = maze
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
    optimal_policy = np.zeros((prob.shape[0], prob.shape[1]), dtype=np.uint16)
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            optimal_policy[i,j] = np.argmax(np.array([prob[i,j,0],prob[i,j,1],prob[i,j,2],prob[i,j,3]]))
    np.savetxt('optimal_policy' + str(iterations) + '.txt', (optimal_policy), fmt="%d")

def save_path(path, iterations):
    np.savetxt('path' + str(iterations) + '.txt', (path), fmt="%d")

def backward_pass(maze, prob, path, steps, reward, alpha, discount_steps, gain):
    counter = 0
    while steps > 0:
        num_valid_moves = valid_moves(maze, path, steps)
        if num_valid_moves > 1:
            # update probabilities
            prob = update_probabilities(maze, prob, path, steps, reward, alpha, discount_steps, counter, num_valid_moves)
            steps -= 1
            counter += 1

            # softmax
            prob = softmax(maze, prob, path, steps, gain)

    return prob

def valid_moves(maze, path, steps):
    num_valid_moves = 0
    if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
        num_valid_moves += 1
    if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
        num_valid_moves += 1
    try:
        if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
            num_valid_moves += 1
    except:
        pass
    try:
        if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
            num_valid_moves += 1
    except:
        pass

    return num_valid_moves

def update_probabilities(maze, prob, path, steps, reward, alpha, discount_steps, counter, num_valid_moves):
    if path[steps,0] == path[steps-1,0] and path[steps,1] == (path[steps-1,1] - 1):
        if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
            prob[path[steps-1,0],path[steps-1,1],1] = prob[path[steps-1,0],path[steps-1,1],1] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],0]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        try:
            if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
                prob[path[steps-1,0],path[steps-1,1],2] = prob[path[steps-1,0],path[steps-1,1],2] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],0]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        try:
            if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
                prob[path[steps-1,0],path[steps-1,1],3] = prob[path[steps-1,0],path[steps-1,1],3] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],0]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        prob[path[steps-1,0],path[steps-1,1],0] = prob[path[steps-1,0],path[steps-1,1],0] + alpha * (1 - prob[path[steps-1,0],path[steps-1,1],0]) * reward * (discount_steps ** counter)
    elif path[steps,0] == (path[steps-1,0] -1) and path[steps,1] == path[steps-1,1]:
        if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
            prob[path[steps-1,0],path[steps-1,1],0] = prob[path[steps-1,0],path[steps-1,1],0] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],1]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        try:
            if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
                prob[path[steps-1,0],path[steps-1,1],2] = prob[path[steps-1,0],path[steps-1,1],2] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],1]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        try:
            if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
                prob[path[steps-1,0],path[steps-1,1],3] = prob[path[steps-1,0],path[steps-1,1],3] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],1]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        prob[path[steps-1,0],path[steps-1,1],1] = prob[path[steps-1,0],path[steps-1,1],1] + alpha * (1 - prob[path[steps-1,0],path[steps-1,1],1]) * reward * (discount_steps ** counter)
    elif path[steps,0] == path[steps-1,0] and path[steps,1] == (path[steps-1,1] + 1):
        if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
            prob[path[steps-1,0],path[steps-1,1],0] = prob[path[steps-1,0],path[steps-1,1],0] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],2]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
            prob[path[steps-1,0],path[steps-1,1],1] = prob[path[steps-1,0],path[steps-1,1],1] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],2]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        try:
            if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
                prob[path[steps-1,0],path[steps-1,1],3] = prob[path[steps-1,0],path[steps-1,1],3] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],2]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        prob[path[steps-1,0],path[steps-1,1],2] = prob[path[steps-1,0],path[steps-1,1],2] + alpha * (1 - prob[path[steps-1,0],path[steps-1,1],2]) * reward * (discount_steps ** counter)
    elif path[steps,0] == (path[steps-1,0] +1) and path[steps,1] == path[steps-1,1]:
        if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
                prob[path[steps-1,0],path[steps-1,1],0] = prob[path[steps-1,0],path[steps-1,1],0] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],3]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
                prob[path[steps-1,0],path[steps-1,1],1] = prob[path[steps-1,0],path[steps-1,1],1] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],3]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        try:
            if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
                prob[path[steps-1,0],path[steps-1,1],2] = prob[path[steps-1,0],path[steps-1,1],2] - alpha * (1 - prob[path[steps-1,0],path[steps-1,1],3]) * reward * (discount_steps ** counter) / (num_valid_moves - 1)
        except:
            pass
        prob[path[steps-1,0],path[steps-1,1],3] = prob[path[steps-1,0],path[steps-1,1],3] + alpha * (1 - prob[path[steps-1,0],path[steps-1,1],3]) * reward * (discount_steps ** counter)

    return prob

def softmax(maze, prob, path, steps, gain):
    softmax_total = 0.0
    if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
        softmax_total += np.exp(gain * prob[path[steps-1,0],path[steps-1,1],0])
    if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
        softmax_total += np.exp(gain * prob[path[steps-1,0],path[steps-1,1],1])
    try:
        if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
            softmax_total += np.exp(gain * prob[path[steps-1,0],path[steps-1,1],2])
    except:
        pass
    try:
        if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
            softmax_total += np.exp(gain * prob[path[steps-1,0],path[steps-1,1],3])
    except:
        pass

    if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
        prob[path[steps-1,0],path[steps-1,1],0] = np.exp(gain * prob[path[steps-1,0],path[steps-1,1],0]) / softmax_total
    if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
        prob[path[steps-1,0],path[steps-1,1],1] = np.exp(gain * prob[path[steps-1,0],path[steps-1,1],1]) / softmax_total
    try:
        if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
            prob[path[steps-1,0],path[steps-1,1],2] = np.exp(gain * prob[path[steps-1,0],path[steps-1,1],2]) / softmax_total
    except:
        pass
    try:
        if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
            prob[path[steps-1,0],path[steps-1,1],3] = np.exp(gain * prob[path[steps-1,0],path[steps-1,1],3]) / softmax_total
    except:
        pass

    return prob

def main(load_maze, change_values, plot_maze, x_row, y_row, prob_not_wall, prob_reward, num_iterations, alpha, discount_steps, gain, gain_factor):
    '''

    :param load_maze: if True, load previously used maze, else create new maze
    :param change_values: if True, change reward values at num_iterations / 2, else maintain current values
    :param plot_maze: if True, display agent position in maze
    :param x_row: Number of rows in the maze
    :param y_row: Number of columns in the maze
    :param prob_not_wall: The probability that a block will not be a wall
    :param prob_reward: The probability that a non-wall block will be a reward
    :param num_iterations: Number of iterations the program runs for.
                           At num_iterations / 2, the gain is reset, and if change_values == True,
                           the program randomly changes the reward values
    :param alpha: Learning rate
    :param discount_steps: Discount factor
    :param gain: Initial trade-off value between exploration and exploitation
                            At num_iterations / 2, the program resets gain
    :param gain_factor: Multiplicative factor for gain
    :return: None

    '''

    # interactive feature of matplotlib
    if plot_maze == True:
        plt.ion()

    # initialize maze
    maze = create_maze(load_maze, x_row, y_row, prob_not_wall, prob_reward)

    # initialize probabilities
    prob = initialize_prob(maze)

    # Begin experiment
    gain2 = gain
    for iterations in range(num_iterations):
        if iterations % 1000 == 0:
            print(iterations)
        if iterations == int(num_iterations/2-1):
            gain = gain2
            maze = update_maze(load_maze, change_values, maze)

        #forward pass
        path, reward, steps = forward_pass(plot_maze, maze, prob)
        if iterations == (num_iterations-1) or iterations == int(num_iterations/2-1):
            save_policy(prob, iterations)
            save_path(path, iterations)

        #backward pass
        prob = backward_pass(maze, prob, path, steps, reward, alpha, discount_steps, gain)
        gain *= gain_factor

if __name__ == "__main__":
    # parameters
    load_maze = True
    change_values = True
    plot_maze = False
    x_row = 9
    y_row = 6
    prob_not_wall = 0.75
    prob_reward = 0.12
    num_iterations = 18000
    alpha = 0.05
    discount_steps = 0.95
    # gain: high values -> exploit, low values -> explore
    gain = 0.05
    # multiplicative factor for the gain, applied after each iteration
    gain_factor = 1.0005

    # Stopwatch
    start_time = time.time()

    # Novel reinforcement learning algorithm
    main(load_maze, change_values, plot_maze, x_row, y_row, prob_not_wall, prob_reward, num_iterations, alpha, discount_steps, gain, gain_factor)

    print("--- %s seconds ---" % (time.time() - start_time))