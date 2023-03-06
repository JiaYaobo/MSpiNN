ks=(2 3)
ns=(500 300)
errs=(0 1 2)


for n in "${ns[@]}"; do
    for e in "${errs[@]}"; do
        for i in {1..50}; do
            for k in {2..3}; do
                if [[ $k -eq 3 ]]
                then
                    python -u train.py --k "$k" --layer_sizes 100 50 --adam_learn_rate 1e-3\
                                    --n_train_obs "$n" --n_epochs 100 --num_p 100 --balance 0.7\
                                    --linear --err_dist "$e" --seed "$i" --round "$i"\
                                    --lasso_param_ratio 0.1 --group_lasso_param 0.27 --ridge_param 0.01 >> logs/MSpINN/train_linear_imbalance2.log
                    wait
                else
                    python -u train.py --k "$k" --layer_sizes 100 50 --adam_learn_rate 1e-3\
                                    --n_train_obs "$n" --n_epochs 100 --num_p 100 --balance 0.7\
                                    --linear --err_dist "$e" --seed "$i" --round "$i"\
                                    --lasso_param_ratio 0.1 --group_lasso_param 0.2 --ridge_param 0.01 >> logs/MSpINN/train_linear_imbalance2.log
                    wait
                fi
            done
        done
    done
done


