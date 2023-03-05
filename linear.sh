# ks=(1 2 3 4)
# ns=(500 300)
# errs=(0 1 2)

# for n in "${ns[@]}"; do
#     for e in "${errs[@]}"; do
#         for i in {1..10}; do
#             for k in {1..4}; do
#                 python -u train.py --k "$k" --layer_sizes 100 50 --adam_learn_rate 1e-3\
#                                 --n_train_obs "$n" --n_epochs 100 --num_p 100\
#                                 --linear --err_dist "$e" --seed "$i" --round "$i"\
#                                 --lasso_param_ratio 0.1 --group_lasso_param 0.3 --ridge_param 0.01 >> logs/MSpINN/train_linear_balance.log
#                 wait
#             done
#         done
#     done
# done


# for i in {1..10}; do
# python -u train.py --k 2 --layer_sizes 100 50 --adam_learn_rate 1e-3\
#                 --n_train_obs 300 --n_epochs 100 --num_p 100\
#                 --linear --err_dist 2 --seed "$i" --round "$i"\
#                 --lasso_param_ratio 0.1 --group_lasso_param 0.5 --ridge_param 0.01 >> logs/MSpINN/train_linear_balance_err2.csv
# wait
# done

python -u train.py --k 2 --layer_sizes 100 50 --adam_learn_rate 1e-3\
                --n_train_obs 300 --n_epochs 100 --num_p 100\
                --linear --err_dist 2 --seed 0 --round 0\
                --lasso_param_ratio 0.1 --group_lasso_param 0.1 --ridge_param 0.01