# ethical_rl

This repo is based off of the MiniGrid project (https://github.com/maximecb/gym-minigrid), which is in turn based off of OpenAI Gym (https://gym.openai.com/)

### 8/31/20
+ Created project and set up structure
+ First pass at implementing dqn.  
+ Can test with: ```python main.py --test_name cartpole_test```

### 9/12/20
+ Added support for creating custom MiniGrid environments.  See ./services/environments/minigrid_test.py
+ Added a launch script for interactive play.  See ./interactive_play.py

### 9/14/20
+ Made dueling dqn
+ Made double dqn
+ Implemented CNN
+ Implented Dueling Double CNN
+ Implemented wrappers
+ Plumbing: 
  + see ```default_config.json``` for an example of available configurations.  These can be a separate test in ```config.json``` or passed in as arguments to ```main.py```.
  + see ```deployment.py```
  
  Testing If Eli can push to master