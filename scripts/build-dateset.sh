echo "[$(date)] start building price indicators dataset."

data_dir='./dataset/raw/price'
save_dir='./dataset/processed/indicators'

if [ ! -d $data_dir ]; then
    echo "price data file does not exist."
    exit
fi
if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

python -u ./data/preprocessing/scripts/indicator.py -i $data_dir -o $save_dir
echo "[$(date)] price indicators dataset is built."

# build news sentiments dataset
echo "[$(date)] start building news sentiments dataset."
data_dir='./dataset/raw/news'
save_dir='./dataset/processed/sentiments'
lexicon=('SenticNet5' 'SenticNet6' 'LMFinance')
lexicon_dir=('None' 'None' './dataset/lexicon/LMFinance')

if [ ! -d $data_dir ]; then
    echo "news data file does not exist."
    exit
fi
if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

for ((i=0; i<${#lexicon[@]}; i++)); do
    echo "python -u ./data/preprocessing/scripts/sentiment.py -i $data_dir -o $save_dir --lexicon ${lexicon[$i]} --lexicon_dir ${lexicon_dir[$i]}"
done
echo "[$(date)] news sentiments dataset is built."