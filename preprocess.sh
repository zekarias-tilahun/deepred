dataset=${1}
root_dir=./data/${dataset}/

if [ ${dataset} = 'lastfm' ] || [ ${dataset} = 'reddit' ] || [ ${dataset} = 'wikipedia' ]; then
    python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate .8 --dev-rate .1
elif [ ${dataset} = 'mooc' ]; then
    python deepred/preprocess.py --root-dir ${root_dir} --temporal --tr-rate .6 --dev-rate .2
else
    python deepred/preprocess.py --root-dir ${root_dir} --tr-rate .6 --dev-rate .1
fi
