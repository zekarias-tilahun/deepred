dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ]; then
    for i in `seq .1 .1 .7` ; do
        python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate ${i} --dev-rate .1
        python deepred/dine.py --root-dir ${root_dir} --epochs 1 --temporal --reg-cof 0.7 --lr .0001 --dropout .8 --nbr-size 100 
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 1 --sfx rate_${i}
    done
elif [ ${dataset} = 'wikipedia' ]; then
    for i in `seq .1 .1 .7` ; do
        python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate ${i} --dev-rate .1
        python deepred/dine.py --root-dir ${root_dir} --epochs 30 --temporal --reg-cof 0.5 --lr .0001 --dropout .7 --nbr-size 50
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 30 --sfx rate_${i}
    done
elif [ ${dataset} = 'reddit' ]; then
    for i in `seq .8 .1 .8` ; do
        python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate ${i} --dev-rate .1
        python deepred/dine.py --root-dir ${root_dir} --epochs 2 --temporal --reg-cof .6 --lr .0001 --dropout .6 --nbr-size 50
        python deepred/next_item_prediction.py --root-dir ${root_dir} --epoch 2 --sfx rate_${i}
    done
elif [ ${dataset} = 'ctd' ]; then
    python deepred/dine.py --root-dir ${root_dir} --epochs 10 --lr 0.0001 --dropout 0.8 --reg-cof 0.7 --nbr-size 200 --dim 128
elif [ ${dataset} = 'sider' ]; then
    python deepred/dine.py --root-dir ${root_dir} --epochs 10 --lr 0.0001 --dropout 0.5 --reg-cof 0.6 --nbr-size 200 --dim 128
fi