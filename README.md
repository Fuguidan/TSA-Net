# TSA-Net

TSA-Net

A single-channel sleep stage classification method. (under review)

## Environment Settings

- Python 3.7
- Cuda 10.1
- Pytorch 1.8.0
- Numpy 1.21.5
- Pandas 1.0.5
- Scikit-learn 1.0.2
- Matplotlib 3.5.3

## How to use

### Prepare the dataset:

Download the Sleep-EDF dataset. We use two datasets: Sleep-EDF-20 and Sleep-EDF-78.

[Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)

### Preprocess the data:

```python
python prepare_physionet.py --data_dir /path/PSG --output_dir data --select_ch "EEG Fpz-Cz"

```

### Train TSA-Net:

```python
python train_preliminary.py --fold_num 20 --np_data_dir ./data --output ./output
```

### Summary TSA-Net:

```python
python summary.py --output ./output
```
