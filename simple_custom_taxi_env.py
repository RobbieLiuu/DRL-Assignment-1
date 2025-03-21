import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from collections import deque
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, min_grid_size=5, max_grid_size=10, fuel_limit=50):
        """
        Custom Taxi environment with a randomly sized grid and obstacles.
        """
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.passenger_loc = None

        self.grid_size = None
        self.stations = []
        self.obstacles = set()
        self.destination = None

    def is_grid_connected(self):
        """Check if all non-obstacle grid cells form a connected component."""
        valid_positions = {(x, y) for x in range(self.grid_size) 
                        for y in range(self.grid_size) 
                        if (x, y) not in self.obstacles}
        
        if not valid_positions:
            return False
        
        start_pos = next(iter(valid_positions))
        visited = {start_pos}
        queue = [start_pos]
        
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y = queue.pop(0)
            
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (new_pos in valid_positions and new_pos not in visited):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return len(visited) == len(valid_positions)

    def reset(self):
        """Reset the environment with a random grid size, stations, and obstacles."""
        self.grid_size = random.randint(self.min_grid_size, self.max_grid_size)
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # 生成所有格子
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        valid_positions = set(all_positions)

        # 隨機選擇不相鄰的四個車站
        self.stations = []
        
        while len(self.stations) < 4:
            pos = random.choice(list(valid_positions))
            self.stations.append(pos)

            # 移除該位置及其周圍上下左右四個格子，確保不相鄰
            x, y = pos
            neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}
            valid_positions -= neighbors

        # 從剩餘的位置選擇計程車初始位置
        available_positions = list(valid_positions)
        if available_positions:
            self.taxi_pos = random.choice(available_positions)
        else:
            # 如果沒有剩餘的位置，重新調用reset方法進行重置
            return self.reset()

        # 從車站中選擇乘客位置與目的地
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        # 隨機生成障礙物，確保地圖連通
        while True:
            self.obstacles = {pos for pos in available_positions if pos != self.taxi_pos and random.random() < 0.15}
            if self.is_grid_connected():
                break

        return self.get_state(), {}
    

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos, action=None, step=None,fuel = None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        # 放置加油站
        station_symbols = ['R', 'G', 'Y', 'B']
        for i, (sx, sy) in enumerate(self.stations):
            grid[sx][sy] = station_symbols[i % len(station_symbols)]

        # 放置障礙物
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'

        # 放置計程車
        tx, ty = taxi_pos  # 直接使用傳入的 taxi_pos
        grid[tx][ty] = '🚖'

        # 印出步驟與狀態資訊
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"station: {self.stations}")
        print(f"Passenger Position: {self.passenger_loc} {'(In Taxi)' if self.passenger_picked_up else ''}")
        print(f"Destination: {self.destination}")
        print(f"Fuel Left: {self.current_fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # 印出地圖
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")