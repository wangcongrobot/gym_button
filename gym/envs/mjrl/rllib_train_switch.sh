rllib train --run PPO --env mjrl_dkitty_switch-v0 --checkpoint-freq 10 --config '{"num_workers": 25, "ignore_worker_failures": true}'

rllib train --run PPO --env mjrl_dkitty_switch_random-v0 --checkpoint-freq 10 --config '{"num_workers": 25, "ignore_worker_failures": true}'