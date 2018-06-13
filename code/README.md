## Install
- PyTorch
- Install the dependencies required

```
pip install -r requirements.txt
```

## Train
First, start visdom.server

```
python -m visdom.server
```

Then, start training

```
# train on gpu0 and store the visualized results in classifier env of visdom
python main.py train --train-data-root=./data/train --use-gpu=True --env=classifier
```

## Test

```
python main.py test --data-root=./data/test --use-gpu=False --batch-size=256
```

