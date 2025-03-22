import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv
import random
import matplotlib.pyplot as plt
import math

action_name = ["Move South", "Move North", "Move East", "Move West", "PICKUP","DROPOFF"]

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

    if phase == 'find passenger':
        dx = (taxi_row - stations_pos[ongoing_pickup_idx][0])
        dy = (taxi_col - stations_pos[ongoing_pickup_idx][1])

        return (dx, dy, phase, passenger_look, obstacle_north, obstacle_south, obstacle_east, obstacle_west)



    elif phase == 'move to destination':
        dx = (taxi_row - stations_pos[ongoing_destination_idx][0])
        dy = (taxi_col - stations_pos[ongoing_destination_idx][1])
        return (dx, dy, phase, destination_look, obstacle_north, obstacle_south, obstacle_east, obstacle_west)




def q_learning( episodes=10000, alpha=0.1, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9999, reward_shaping=True,
                              q_table=None, debug=False):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Implement Tabular Q-learning with Reward Shaping
    - Modify reward shaping to accelerate learning.
    - Adjust epsilon decay to ensure sufficient exploration.
    - Ensure the agent learns the full sequence: "pick up key → open door → reach goal".
    """

    random_grid_size=random.randint(5, 10) 
    env = SimpleTaxiEnv(random_grid_size, fuel_limit=5000)
    q_table = {}

    rewards_per_episode = []
    epsilon = epsilon_start

    stage_epsilon = {'find passenger': epsilon_start, 'move to destination':epsilon_start}
    for episode in range(episodes):
        while True:  # Retry reset on failure
            try:
                obs, _ = env.reset()
                state = env.get_state()
                break
            except Exception as e:
                print(f"Error resetting environment in episode {episode + 1}: {e}")
                continue


        stations_pos = [(state[2],state[3]),(state[4],state[5]),(state[6],state[7]),(state[8],state[9])]
        ongoing_pickup_idx = 0
        ongoing_destination_idx = 0
        correct_pickup_idx = -1
        correct_destination_idx = -1

        total_reward = 0
        fuel_status = False
        phase = 'find passenger'
        first_time_find_passenger = 0



        while fuel_status == False:
            
            # TODO: Initialize the state in the Q-table if not already present.
            q_state = state_to_q_state(state,phase,ongoing_pickup_idx,ongoing_destination_idx)
            if q_state not in q_table:
                q_table[q_state] = np.zeros(len(action_name))
            
            # TODO: Implement ε-greedy policy for action selection.
            if np.random.rand() < stage_epsilon[phase]:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                action = np.argmax(q_table[q_state])



            # Execute the selected action.
            obs, reward, fuel_status, _ = env.step(action)
            next_state = env.get_state()

            # Implement reward shaping.
            shaped_reward = 0

            if phase == 'find passenger':
                taxi_pos = (state[0],state[1])
                distance_to_ongoing_pickup = abs(taxi_pos[0]-stations_pos[ongoing_pickup_idx ][0]) + abs(taxi_pos[1]-stations_pos[ongoing_pickup_idx ][1])
                passenger_look = state[-2]
                
                if distance_to_ongoing_pickup <= 1 and passenger_look == False: #went to wrong station
                    ongoing_pickup_idx = (ongoing_pickup_idx + 1)%4
                    if action_name[action] == "PICKUP" or action_name[action] == "DROPOFF":
                        shaped_reward -= 10

                elif distance_to_ongoing_pickup == 0 and passenger_look == True : # right pickup point
                    if action_name[action] == "PICKUP":
                        phase == 'move to destination'
                        if first_time_find_passenger == 0:
                            shaped_reward += 25
                            correct_pickup_idx = ongoing_pickup_idx
                            first_time_find_passenger  = 1
                        if correct_pickup_idx == ongoing_destination_idx:
                            ongoing_destination_idx = (ongoing_destination_idx + 1)%4
                    else:
                        shaped_reward -= 10

                    
            elif phase == 'move to destination':
                taxi_pos = (state[0],state[1])
                distance_to_ongoing_destination = abs(taxi_pos[0]-stations_pos[ongoing_destination_idx ][0]) + abs(taxi_pos[1]-stations_pos[ongoing_destination_idx][1])
                destination_look = state[-1]

                if distance_to_ongoing_destination <= 1 and destination_look == False: #went to wrong destination
                    ongoing_destination_idx = (ongoing_destination_idx + 1)%4
                    if correct_pickup_idx == ongoing_destination_idx:
                        ongoing_destination_idx = (ongoing_destination_idx + 1)%4
                    if action_name[action] == "PICKUP":
                        shaped_reward -= 10
                    elif action_name[action] == "DROPOFF":
                        shaped_reward -= 50
                        phase = 'find passenger'

                
                elif distance_to_ongoing_destination == 0 and destination_look == True: # right dropoff point
                    if action_name[action] == "DROPOFF":
                        shaped_reward += 50
                        correct_destination_idx = ongoing_destination_idx
                    else:
                        shaped_reward -= 10



            obstacle_north = state[-6]
            obstacle_south = state[-5]
            obstacle_east = state[-4]
            obstacle_west = state[-3]

            if obstacle_north == 1 and action_name[action] == "Move North":
                shaped_reward -= 3
            elif obstacle_south == 1 and action_name[action] == "Move South":
                shaped_reward -= 3
            elif obstacle_east == 1 and action_name[action] == "Move East":
                shaped_reward -= 3
            elif obstacle_west == 1 and action_name[action] == "Move West":
                shaped_reward -= 3

            # Update total reward.
            reward += shaped_reward
            total_reward += reward

            # TODO: Initialize the next state in the Q-table if not already present.
            next_q_state = state_to_q_state(state,phase,ongoing_pickup_idx,ongoing_destination_idx)
            if next_q_state not in q_table:
                q_table[next_q_state] = np.zeros(len(action_name))

            # Q-learning update
            q_table[q_state][action] += alpha * (reward + gamma * np.max(q_table[next_q_state]) - q_table[q_state][action])
            state = next_state



        rewards_per_episode.append(total_reward)
        stage_epsilon['find passenger'] = max(epsilon_end, stage_epsilon["find passenger"] * decay_rate)
        if phase == 'move to destination':
            stage_epsilon['move to destination'] = max(epsilon_end, stage_epsilon["move to destination"] * decay_rate)


        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {stage_epsilon['find passenger']:.4f} , {stage_epsilon['move to destination']:.4f} ")

    return q_table, rewards_per_episode


if __name__ == "__main__":
    q_table, rewards_per_episode = q_learning(episodes=250000,decay_rate=0.99999)
    with open("q_table_no_abs_w_obstacle.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("q table saved")


    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward Shaping Training Progress")
    plt.savefig("test.png")  # Save as PNG
    plt.close()
