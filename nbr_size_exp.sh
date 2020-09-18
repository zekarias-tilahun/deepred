dataset=${1}
root_dir=./data/${dataset}/
nbr_sizes=(50 100 150 200)
python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate .8 --dev-rate .1
if [ ${dataset} = 'lastfm' ]; then
    for nbr_size in "${nbr_sizes[@]}"; do
        python deepred/deepred.py --root-dir ${root_dir} --epochs 1 --temporal --reg-cof 0.7 --lr .0001 --dropout .8 --nbr-size ${nbr_size}
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 1 --nbr-size ${nbr_size} --sfx nbr_size_${nbr_size}
    done
elif [ ${dataset} = 'wikipedia' ]; then
    for nbr_size in "${nbr_sizes[@]}"; do
        python deepred/deepred.py --root-dir ${root_dir} --epochs 20 --temporal --reg-cof 0.5 --lr .0001 --dropout .7 --nbr-size ${nbr_size}
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 20 --nbr-size ${nbr_size} --sfx nbr_size_${nbr_size}
    done
elif [ ${dataset} = 'reddit' ]; then
    for nbr_size in "${nbr_sizes[@]}"; do
        python deepred/deepred.py --root-dir ${root_dir} --epochs 2 --temporal --reg-cof .6 --lr .0001 --dropout .6 --nbr-size ${nbr_size}
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 2 --nbr-size ${nbr_size} --sfx nbr_size_${nbr_size}
    done
elif [ ${dataset} = 'ctd' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --lr 0.0001 --dropout 0.8 --reg-cof 0.7 --nbr-size 200 --dim 128
elif [ ${dataset} = 'sider' ]; then
    python deepred/deepred.py --root-dir ${root_dir} --epochs 10 --lr 0.0001 --dropout 0.5 --reg-cof 0.6 --nbr-size 200 --dim 128
fi