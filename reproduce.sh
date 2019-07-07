
# Discriminative Results
python train.py 2>&1 | tee discriminative.log

# Generative Results
python train.py 2>&1 --lr 0.005 --hidden-units 6000| tee generative.log


# Hybrid Results
python train.py 2>&1 --lr 0.05 --hidden-units 1500 | tee hybrid.log

