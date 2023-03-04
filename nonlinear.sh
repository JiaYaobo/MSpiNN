python -u train.py --k 2 --layer_sizes 50 80 30  --adam_learn_rate 1e-2\
                   --n_train_obs 300 --n_epochs 300 --num_p 50\
                   --err_dist 1 --seed 1\
                   --lasso_param_ratio 0.01 --group_lasso_param 0.02 --ridge_param 0.01