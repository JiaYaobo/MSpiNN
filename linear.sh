python -u train.py --k 2 --layer_sizes 50 30  --adam_learn_rate 1e-3\
                   --n_train_obs 300 --n_epochs 300 --num_p 100\
                   --linear --err_dist 0 --seed 520\
                   --lasso_param_ratio 0.1 --group_lasso_param 0.08 --ridge_param 0.01