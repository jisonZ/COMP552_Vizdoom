# COMP552_Vizdoom
COMP552 final project

## Our Trained Models are available at
https://drive.google.com/file/d/14PPffTA1F_oTl6EMz0RJs7i0NS5rxsnF/view?usp=sharing

## train_baseline3.py
train PPO and RPPO agent under single enviroment.

## train_baseline3_multiEnv.py
train PPO and RPPO agent under vectorized enviroment and scaled reward.

## CSV_graph_gen.py
generate training graph from monitor CSV file.

## vizdoom_train_PPO_and_Recurrent_PPO.ipynb
Train and the PPO and Recurrent PPO models on Colab and evaluate checkpoints.

## evaluate_PPO_model.py
Evaluate a checkpoint of the PPO model on 10 episodes of defend the center.

## evaluate_recurrent_PPO_model.py
Evaluate a checkpoint of the Recurrent PPO model on 10 episodes of defend the center.

## play_as_human.py
Run to play the defend the center scenario. Press arrow keys to pan left and right, and Ctrl to shoot.

## Dependencies
Install the following packages-
```
pip install vizdoom[gym]
pip install stable-baselines3
pip install sb3-contrib
```

