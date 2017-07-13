# Word-level language modeling RNN

This repo is adapt from [PyTorch Word Language Model]() example with a model wrapper for any model

```bash
python main.py --cuda --epochs 6        # Train a LSTM on PTB with CUDA, reaching perplexity of 117.61
python main.py --cuda --epochs 6 --tied # Train a tied LSTM on PTB with CUDA, reaching perplexity of 110.44
python main.py --cuda --tied            # Train a tied LSTM on PTB with CUDA for 40 epochs, reaching perplexity of 87.17
python generate.py                      # Generate samples from the trained LSTM model.
```

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluted against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        humber of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 300 --nhid 650 --dropout 0.5 --epochs 40

```