#!/bin/bash
err=(0 1 2)
ns=(300 500)
balance=(0.5 0.7)


for n in "${ns[@]}";do
    for d in "${err[@]}"; do
        for b in "${balance[@]}"; do
                python -u make_dataset.py --balance "$b"  --err_dist "$d" --n_train_obs "$n" 
                wait
        done
    done
done

wait

for n in "${ns[@]}";do
    for d in "${err[@]}"; do
        for b in "${balance[@]}"; do
                python -u make_dataset.py --linear --balance "$b"  --err_dist "$d" --n_train_obs "$n" 
                wait
        done
    done
done

echo "Dataset Created Finished"