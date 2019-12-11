# polar-position-embedding demo

## tow step to run the Text Classification demo:
1. build the word dict file

select dataset in utils.py and run it:
```python
python utils.py
```

2. train the polar Transformer model:
```python
python train.py -b=32
```

## some tips
1. We build our model based on Pytorch 1.1
2. The details of the model configs can be found in the 'cof' in train.py.
