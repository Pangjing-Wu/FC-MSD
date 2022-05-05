mode='ms-clf'

# run baseline.
if [[ $mode == 'baseline' ]]; then
    baseline_logdir='./logs/classification/baseline'
    if [ ! -d $baseline_logdir ]; then
        mkdir -p $baseline_logdir
    fi

    lexicons=('SenticNet5' 'SenticNet6' 'LMFinance')
    clfs=('SVM' 'GBDT' 'AdaBoost')
    stocks='dataset/stocklist.txt'

    cat $stocks| while read stock; do
        for lexicon in ${lexicons[@]}; do
            for clf in ${clfs[@]}; do
                nohup python -u ./main.py -s $stock --lexicon $lexicon --clf $clf 2>&1 > $baseline_logdir/$stock-$lexicon-$clf.log &
                sleep 1
            done
        done
        wait
    done

# run ms-clf.
else
    logdir='./logs/classification/ms-clf'
    if [ ! -d $logdir ]; then
        mkdir -p $logdir
    fi

    lexicons=('SenticNet5' 'SenticNet6' 'LMFinance')
    clfs=('SVM' 'GBDT' 'AdaBoost')
    clses=('KMeans' 'Hierarchy')
    enses=('GBDT' 'AdaBoost')
    stocks='dataset/stocklist.txt'

    cat $stocks| while read stock; do
        for lexicon in ${lexicons[@]}; do
            for clf in ${clfs[@]}; do
                for cls in ${clses[@]}; do
                    for ens in ${enses[@]}; do
                        nohup python -u ./main.py -s $stock --lexicon $lexicon --clf $clf --cls $cls --ens $ens 2>&1 > $logdir/$stock-$lexicon-$clf-$cls-$ens.log &
                        sleep 1
                    done
                done
            done
        done
        wait
    done
fi