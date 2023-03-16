for i in {0..9}; do
    python -u linear_ccle.py --k 2 --layer_sizes 100 --n_epochs 200 --num_p 100 --seed "$i"\
                            --ridge_param 0.01 --group_lasso_param 0.2 --lasso_param_ratio 0.1
done