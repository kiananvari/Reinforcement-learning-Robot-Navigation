# Reinforcement Learning for Robot Navigation in Webots

This repository contains the implementation code and simulated environment for training robots to autonomously navigate and reach a goal while avoiding obstacles using Proximal Policy Optimization (PPO) reinforcement learning algorithm. The project utilizes Python programming language and the Webots robotics simulation environment. The robot platform used for experimentation is the Pioneer 3-DX.

## Overview

The goal of this project is to train a robot to navigate a simulated environment, reaching a designated goal while avoiding obstacles. The implemented solution utilizes different sensor combinations for effective navigation.

In the first section, the robot utilizes GPS and Distance Sensors to navigate towards the goal. The GPS sensor provides positional information, while the Distance Sensors detect obstacles, enabling the robot to avoid collisions and make progress towards the goal.

In the next section, the robot incorporates a Camera sensor in addition to GPS. This combination allows the robot to perceive the environment visually and extract useful information to avoid obstacles and reach the goal successfully. By leveraging the camera's visual inputs, the robot gains a more comprehensive understanding of its surroundings and can make informed decisions to navigate safely.

The repository includes the implementation code and simulated environment for both sections, allowing for a comparison of the two approaches and their respective performance in robot navigation.
## Features

- Reinforcement Learning
- PPO
- Robot navigation


## Documentation

You can see the description of the implementation method in the following file:
[Click Here](https://github.com/kiananvari/Reinforcement-learning-Robot-Navigation/raw/main/Documentation.pdf)


## Results

Navigation Using Distance Sensor and GPS from Start Point:

![App Screenshot](https://github.com/kiananvari/Reinforcement-learning-Robot-Navigation/raw/main/gifs/1-ORG.gif)
Navigation Using Distance Sensor and GPS from Another Start Point (Without Training Again):

![App Screenshot](https://github.com/kiananvari/Reinforcement-learning-Robot-Navigation/raw/main/gifs/1-MOVED.gif)

Navigation Using Camera Sensor and GPS from Start Point:

![App Screenshot](https://github.com/kiananvari/Reinforcement-learning-Robot-Navigation/raw/main/gifs/2.gif)

Training Phase Metrics Plots:

![App Screenshot](https://github.com/kiananvari/Reinforcement-learning-Robot-Navigation/raw/main/plots.png)

