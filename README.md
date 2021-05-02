# DETM

# Prerequsites

To create a correct conda environment, please use the provided `environment.yml`.
```
conda env create -f environment.yml
conda activate cdtm
```

## Training
```
python main.py --dataset DATASET_NAME --data_path datasets/processed/ --emb_path datasets/processed/embedding.json --train_embeddings 0  --num_topics 10 --seed 9999 --epochs 100 --mode train
```
example:
```
python main.py --dataset brazil_news --data_path datasets/processed/ --emb_path datasets/processed/embedding.json --train_embeddings 1  --num_topics 10 --lr 0.001 --lr_factor 1.0 --seed 9999 --epochs 100 --mode train
```

## Evaluation
```
python main.py --dataset DATASET_NAME --data_path datasets/processed/ --emb_path datasets/processed/embedding.json --train_embeddings 0  --num_topics 10 --seed 9999 --epochs 100 --mode eval --load_from ./results/MODE_FILE_NAME
```

## Citation
```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```


