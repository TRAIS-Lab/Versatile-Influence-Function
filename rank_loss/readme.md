## Train full model
```bash
python main.py --dataset Delicious --loss attn_rank --device cuda --emb_dim 2 --ce_average --svd --remove_label -1 --seed 0
python main.py --dataset Mediamill2 --loss attn_rank --device cuda --emb_dim 2 --ce_average --svd --remove_label -1 --seed 0
```

## VIF score
```bash
python score_mediamill.py --dataset Mediamill2 --loss attn_rank --device cuda:1 --emb_dim 2 --ce_average --batch_size 128 --svd
python score_delicious.py --dataset Delicious --loss attn_rank --device cuda:1 --emb_dim 2 --ce_average --batch_size 128 --svd
```

## brute-force training
```bash
python main.py --dataset Delicious --loss attn_rank --device cuda --emb_dim 2 --ce_average --svd --remove_label <0-99> --seed 0
python main.py --dataset Mediamill2 --loss attn_rank --device cuda --emb_dim 2 --ce_average --svd --remove_label <0-100> --seed 0
```

## GT calculation
```bash
python gt_mediamill.py --dataset Mediamill2 --loss attn_rank --device cuda:1 --emb_dim 2 --ce_average --batch_size 512 --svd
python gt_delicious.py --dataset Delicious --loss attn_rank --device cuda:1 --emb_dim 2 --ce_average --batch_size 512 --svd
```