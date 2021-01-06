Code repo for paper:
## Geometric Scattering Attention Networks
* https://arxiv.org/abs/2010.15010

Run (default: Cora)
```
python train.py
```

For wikics(running for the 20 splits and take mean accuracy) <br>
the splits are randomly selected, see Sec 3.3 in https://arxiv.org/abs/2010.15010
```
for i in $(seq 0 1 19)
do
echo 'Data Spilt:======='
echo $i
python -u WikiCS_ATT.py --dropout 0.3 --epochs 600 --smoo 0.3 --hid 20 --nheads 20 --weight_decay 1e-3 --data_spilt $i
done
```

Requirement:
* Pytorch Geometric
* Pytorch

