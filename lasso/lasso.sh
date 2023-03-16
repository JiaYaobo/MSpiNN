python -u lasso.py --k 6 --layer_sizes 100 --adam_learn_rate 1e-3\
                --n_train_obs 300 --n_epochs 200 --num_p 100\
                --linear --err_dist 1 --seed 666 --round 0\
                --lasso_param_ratio 0.1 --group_lasso_param 0.1 --ridge_param 0.01