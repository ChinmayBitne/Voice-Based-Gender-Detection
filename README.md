# Voice-Based-Gender-Detection

Gender Recognition Using Real Time Voice Input or by using file input.
## Requirements
- Numpy
- Pandas
- PyAudio
- Librosa
- TensorFlow 
- Scikit-learn
- Streamlit

### Required libraries:
    pip install -r requirements.txt

## Dataset
Before using the data file for data preparation, extract it.
[Mozilla's Common Voice](https://www.kaggle.com/mozillaorg/common-voice)

- [`preparation.py`](preparation.py) is used if you want to download the dataset and extract the features files (.npy files) manually.

## Training
Customize your model in [`utils.py`](utils.py) file with the create_model() function :

    python train.py

## Demo Website
  - Link:
    [Click](..)
