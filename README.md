# HENCE
The official implementation of AAAI 2024 paper: Estimating On-road Transportation Carbon Emissions from Open Data of Road Network and Origin-destination Flow Data

### Model Training
To train our model:
```
python main.py --lr 6e-3 --epochs 1000 --batch_size 16 --patience 20 --scale 0 --num_heads 2 --pretrain_epoch 100 --attention 1 --mapping_dim 8

