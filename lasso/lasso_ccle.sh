python -u lasso_ccle.py --k 2 --layer_sizes 100 --adam_learn_rate 1e-3\
                --n_epochs 300 --num_p 100\
                --linear --err_dist 1 --seed 0 --round 0\
                --lasso_param_ratio 0.1 --group_lasso_param 0.3 --ridge_param 0.01 >> logs/ccle/lasso.csv