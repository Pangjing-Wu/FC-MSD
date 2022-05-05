logdir='logs/grid-search/classifier'
if [ ! -d $logdir ]; then
    mkdir -p $logdir
fi

lexicons=('SenticNet5' 'SenticNet6' 'LMFinance')
clfs=('SVM' 'GBDT' 'AdaBoost')
stocks='dataset/stocklist.txt'

cat $stocks| while read stock; do
    for lexicon in ${lexicons[@]}; do
        for clf in ${clfs[@]}; do
            nohup python -u ./grid-search/classifier.py -v -s $stock --lexicon $lexicon --clf $clf 2>&1 > "$logdir/$stock-$lexicon-$clf.log" &
            sleep 1
        done
    done
    wait
done