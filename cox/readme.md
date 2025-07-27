## Train full model
```bash
python train_cox.py --remove_index -1 --dataset <metabric/support>
```

## VIF score
```bash
python score_cox.py --dataset <metabric/support>
```

## brute-force training
```bash
python train_cox.py --remove_index <0-1216/0-5676> --dataset <metabric/support>
```

## GT calculation
```bash
python gt_cox.py --dataset <metabric/support>
```
