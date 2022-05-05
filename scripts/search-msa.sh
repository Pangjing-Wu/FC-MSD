logdir='logs/grid-search/msa'
if [ ! -d $logdir ]; then
    mkdir -p $logdir
fi

lexicons=('SenticNet5' 'SenticNet6' 'LMFinance')
enses=('GBDT' 'AdaBoost')
clses=('KMeans' 'Hierarchy')
stocks='dataset/stocklist.txt'

cat $stocks| while read stock; do
    for lexicon in ${lexicons[@]}; do
        for cls in ${clses[@]}; do
            for ens in ${enses[@]}; do
                nohup python -u ./grid-search/msa.py -v -s $stock --lexicon $lexicon --cls $cls --ens $ens 2>&1 > "$logdir/$stock-$lexicon-$cls-$ens.log" &
                sleep 1
            done
        done
    done
done