nohup  python run_re2.py -env="Ant-v2" -disable_cuda -OFF_TYPE=1 -pr=64 -pop_size=5 -prob_reset_and_sup=0.05  -time_steps=200 -theta=0.5 -frac=0.7  -gamma=0.99 -TD3_noise=0.2 -EA -RL -K=1 -state_alpha=0.0 -actor_alpha=1.0 -EA_actor_alpha=1.0 -tau=0.005 -seed=1 -logdir="./logs" > ./logs/xxx.log 2>&1 &


python train_ERLRE_DRAMSys.py  -OFF_TYPE=1 -pr=64 -pop_size=5 -prob_reset_and_sup=0.05  -time_steps=200 -theta=0.5 -frac=0.7  -gamma=0.99 -TD3_noise=0.2 -EA -RL -K=1 -state_alpha=0.0 -actor_alpha=1.0 -EA_actor_alpha=1.0 -tau=0.005 -seed=1 -logdir="./logs" > ./logs/xxx.log 2>&1 &


