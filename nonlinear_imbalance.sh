ks=(1 2 3 4)
ns=(500 300)
errs=(0 1 2)

for n in "${ns[@]}"; do
    for e in "${errs[@]}"; do
        for i in {1..10}; do
            for k in {1..4}; do
                python -u train.py --k "$k" --layer_sizes 100 50 --adam_learn_rate 1e-3\
                                --n_train_obs "$n" --n_epochs 100 --num_p 100\
                                --err_dist "$e" --seed "$i" --round "$i" --balance 0.7\
                                --lasso_param_ratio 0.1 --group_lasso_param 0.2 --ridge_param 0.01 >> logs/MSpINN/train_nonlinear_imbalance.log
                wait
            done
        done
    done
done