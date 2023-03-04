for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 0 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_balance_300_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 1 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_balance_300_mlp_err_1.csv
    wait
done


for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 2 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_balance_300_mlp_err_2.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 0 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_balance_500_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 1 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_balance_500_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.5 --err_dist 2 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_balance_500_mlp_err_2.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 0 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_imbalance_300_ensembles_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 1 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_imbalance_300_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 2 --n_train_obs 300 --seed "$i" >> logs/mlp/nonlinear_imbalance_300_mlp_err_2.csv
    wait
done


for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 0 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_imbalance_500_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 1 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_imbalance_500_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --balance 0.7 --err_dist 2 --n_train_obs 500 --seed "$i" >> logs/mlp/nonlinear_imbalance_500_mlp_err_2.csv
    wait
done

###### 


for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 0 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_balance_300_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 1 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_balance_300_mlp_err_1.csv
    wait
done


for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 2 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_balance_300_mlp_err_2.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 0 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_balance_500_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 1 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_balance_500_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.5 --err_dist 2 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_balance_500_mlp_err_2.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 0 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_imbalance_300_ensembles_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 1 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_imbalance_300_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 2 --n_train_obs 300 --seed "$i" >> logs/mlp/linear_imbalance_300_mlp_err_2.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 0 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_imbalance_500_mlp_err_0.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 1 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_imbalance_500_mlp_err_1.csv
    wait
done

for i in {1..10}; do
    python -u mlp.py --linear --balance 0.7 --err_dist 2 --n_train_obs 500 --seed "$i" >> logs/mlp/linear_imbalance_500_mlp_err_2.csv
    wait
done