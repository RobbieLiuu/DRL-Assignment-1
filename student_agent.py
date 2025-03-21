# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from simple_custom_taxi_env import SimpleTaxiEnv
import matplotlib.pyplot as plt
import math

def get_random_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)


action_name = ["Move South", "Move North", "Move East", "Move West", "PICKUP","DROPOFF"]
phase = "find passenger"
ongoing_pickup_idx = 0
ongoing_destination_idx = 0
delta = 0.1
correct_pickup_idx = -1

def state_to_q_state(state , phase, ongoing_pickup_idx, ongoing_destination_idx):

    taxi_row = state[0]
    taxi_col = state[1]
    obstacle_north = state[-6]
    obstacle_south = state[-5]
    obstacle_east = state[-4]
    obstacle_west = state[-3]
    passenger_look = state[-2]
    destination_look = state[-1]
    stations_pos = [(state[2],state[3]),(state[4],state[5]),(state[6],state[7]),(state[8],state[9])]

    if phase == "find passenger":
        dx = abs(taxi_row - stations_pos[ongoing_pickup_idx][0])
        dy = abs(taxi_col - stations_pos[ongoing_pickup_idx][1])

        return (dx, dy, phase, passenger_look, obstacle_north, obstacle_south, obstacle_east, obstacle_west)
        #return (dx, dy, phase, passenger_look)



    elif phase == "move to destination":
        dx = abs(taxi_row - stations_pos[ongoing_destination_idx][0])
        dy = abs(taxi_col - stations_pos[ongoing_destination_idx][1])
        return (dx, dy, phase, destination_look, obstacle_north, obstacle_south, obstacle_east, obstacle_west)
        #return (dx, dy, phase, destination_look)
    

def get_action(obs):
    global action_name
    global Q_table
    global phase
    global ongoing_pickup_idx
    global ongoing_destination_idx
    global delta
    global correct_pickup_idx

    state = obs

    q_state = state_to_q_state(state, phase, ongoing_pickup_idx, ongoing_destination_idx)
    stations_pos = [(state[2],state[3]),(state[4],state[5]),(state[6],state[7]),(state[8],state[9])]


    if q_state not in Q_table:
        action = random.choice([0, 1, 2, 3, 4, 5])
    else:
        if np.random.rand() < delta:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(Q_table[q_state])
         


    # change variable value (after this action)
    if phase == "find passenger":
            taxi_pos = (state[0],state[1])
            distance_to_ongoing_pickup = abs(taxi_pos[0]-stations_pos[ongoing_pickup_idx ][0]) + abs(taxi_pos[1]-stations_pos[ongoing_pickup_idx ][1])
            passenger_look = state[-2]
                
            if distance_to_ongoing_pickup <= 1 and passenger_look == False: #went to wrong station
                    ongoing_pickup_idx = (ongoing_pickup_idx + 1)%4
       #             if action_name[action] == "PICKUP" or action_name[action] == "DROPOFF":
        #                shaped_reward -= 10

            elif distance_to_ongoing_pickup == 0 and passenger_look == True: # right pickup point
                    if action_name[action] == "PICKUP":
                        phase == "move to destination"
                     #   shaped_reward += 25
                        correct_pickup_idx = ongoing_pickup_idx
                        if correct_pickup_idx == ongoing_destination_idx:
                            ongoing_destination_idx = (ongoing_destination_idx + 1)%4
               #     else:
                #        shaped_reward -= 10

                    
    elif phase == "move to destination":
                taxi_pos = (state[0],state[1])
                distance_to_ongoing_destination = abs(taxi_pos[0]-stations_pos[ongoing_destination_idx ][0]) + abs(taxi_pos[1]-stations_pos[ongoing_destination_idx][1])
                destination_look = state[-1]

                if distance_to_ongoing_destination <= 1 and passenger_look == False: #went to wrong destination
                    ongoing_destination_idx = (ongoing_destination_idx + 1)%4
                    if correct_pickup_idx == ongoing_destination_idx:
                        ongoing_destination_idx = (ongoing_destination_idx + 1)%4
                 #   if action_name[action] == "PICKUP":
                  #      shaped_reward -= 10
                   # elif action_name[action] == "DROPOFF":
                    #    shaped_reward -= 50

                
          #      elif distance_to_ongoing_destination == 0 and passenger_look == True: # right dropoff point
           #         if action_name[action] == "DROPOFF":
                 #       shaped_reward += 50
            #            correct_destination_idx = ongoing_destination_idx
               #     else:
                #        shaped_reward -= 10

    return action
