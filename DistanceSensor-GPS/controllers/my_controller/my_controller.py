# Add Webots controlling libraries
from controller import Robot
from controller import Supervisor


# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Stable_baselines3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment(gym.Env, Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters
        self.max_speed = 1.5 # Maximum Angular speed in rad/s
        self.start_coordinate = np.array([-2.60, -2.96])
        self.destination_coordinate = np.array([-0.03, 2.72]) # Target (Goal) position
        self.reach_threshold = 0.06 # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])
        
        
        # Activate Devices
        #~~ 1) Wheel Sensors
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        #~~ 2) GPS Sensor
        sampling_period = 1 # in ms
        self.gps = robot.getDevice("gps")
        self.gps.enable(sampling_period)
        
        #~~ 3) Enable Touch Sensor
        self.touch = robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)
              
        # List of all available sensors
        available_devices = list(robot.devices.keys())
        # Filter sensors name that contain 'so'
        filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

        #Space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.max_steps = 500

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200) # take some dummy steps in environment for initialization
        
        
        # Create dictionary from all available distance sensors and keep min and max of from total values
        self.max_sensor = 0
        self.min_sensor = 0
        self.dist_sensors = {}
        for i in filtered_list:    
            self.dist_sensors[i] = robot.getDevice(i)
            self.dist_sensors[i].enable(sampling_period)
            self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)    
            self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)
           
            
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
        

    def get_distance_to_goal(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
        
    def get_distance_to_start(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_start = np.linalg.norm(self.start_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_start, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
    
    def get_sensor_data(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.
        sensor_data = []
        for z in self.dist_sensors:
            sensor_data.append(self.dist_sensors[z].value)  
            
        sensor_data = np.array(sensor_data)
        normalized_sensor_data = self.normalizer(sensor_data, self.min_sensor, self.max_sensor)
        
        return normalized_sensor_data
        
        
    def get_current_position(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.

            
        position = self.gps.getValues()[0:2]
        position = np.array(position)

        normalized_current_position = self.normalizer(position, -4, +4)
        
        return normalized_current_position
    
    def get_observations(self):
        # """
        # Obtains and returns the normalized sensor data, current distance to the goal, and current position of the robot.
    
        # Returns:
        # - numpy.ndarray: State vector representing distance to goal, distance sensor values, and current position.
        # """
    
        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalized_current_distance = np.array([self.get_distance_to_goal()], dtype=np.float32)
        normalized_current_position = np.array(self.get_current_position(), dtype=np.float32)

        state_vector = np.concatenate([normalized_current_distance, normalized_sensor_data, normalized_current_position], axis=0)

        return state_vector
    
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """

        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations(), {}


    def step(self, action):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        self.apply_action(action)
        step_reward, done = self.get_reward()
        state = self.get_observations()
        # Time-based termination condition
        if (int(self.getTime()) + 1) % self.max_steps == 0:
            done = True
        none = 0
        return state, step_reward, done, none, {}
        

    
    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        
        done = False
        reward = 0
        
        normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = self.get_distance_to_goal()
        normalized_start_distance = self.get_distance_to_start()
        
        normalized_current_distance *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100
                
            
         # (1) Reward according to distance 
        if normalized_current_distance < 42:
            if normalized_current_distance < 10:
                growth_factor = 5
                A = 2.5
            elif normalized_current_distance < 25:
                growth_factor = 4
                A = 1.5
            elif normalized_current_distance < 37:
                growth_factor = 2.5
                A = 1.2
            else:
                growth_factor = 1.2
                A = 0.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))
            
        else: 
            reward += -normalized_current_distance / 100
            
            
        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        if normalized_current_distance < reach_threshold:
            # Reward for finishing the task
            done = True
            reward += 50
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            done = True
            reward -= 5
            
            
        # (3) Punish if close to obstacles
        elif np.any(normalized_sensor_data[normalized_sensor_data > self.obstacle_threshold]):
            reward -= 0.0001

        return reward, done


    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        if action == 0: # move forward
            # print("forward")
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1: # turn right
            # print("right")
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
        elif action == 2: # turn left
            # print("left")
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        
        robot.step(500)

        
        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)           
    

class Agent_FUNCTION():
    def __init__(self, save_path, num_episodes):
        self.save_path = save_path
        self.num_episodes = num_episodes

        self.env = Environment()

        #PPO
        self.policy_network = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log="./results_tensorboard/")#.learn(total_timesteps=1000)
        
    
    def save(self):
        print(self.save_path ,"PPO-Best")
        self.policy_network.save(self.save_path + "PPO-Best")

    def load(self):
        print(self.save_path+"PPO-Best")
        self.policy_network = PPO.load(self.save_path+"PPO-Best")


    def train(self) :
        start_time = time.time()
        reward_history = []

        self.policy_network.learn(total_timesteps=(num_episodes*100))
 
        self.save()

 
    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            # state = self.env.reset()
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                # print("state: ", state)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                # print("reward: ", reward)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()
        
            
            
if __name__ == '__main__':
    # Configs
    save_path = './results/'   
    train_mode = False 
    num_episodes = 1000 if train_mode else 10


    env = Environment()

    agent = Agent_FUNCTION(save_path, num_episodes)


    if train_mode:
        # Initialize Training
        agent.train()
    else:
        # Load PPO
        agent.load()
        # Test
        agent.test()
