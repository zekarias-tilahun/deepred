dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 3
elif [ ${dataset} = 'wikipedia' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 1
elif [ ${dataset} = 'reddit' ]; then
    python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 2
elif [ ${dataset} = 'ctd' ]; then
    python deepred/interaction_prediction.py --root-dir ${root_dir} --epoch 3
elif [ ${dataset} = 'sider' ]; then
    python deepred/interaction_prediction.py --root-dir ${root_dir} --epoch 3
fi
