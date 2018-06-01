# google-landmarks
My solution for the Google Landmark challenges on Kaggle: https://www.kaggle.com/c/landmark-recognition-challenge/ and https://www.kaggle.com/c/landmark-retrieval-challenge/. It was enough to secure a place in Top-40.

## How it works.
An ensemble of four networks:
1. ResNet50
2. DenseNet121
3. DenseNet169
4. SE-ResNext101_32x3d

Then non-landmark filter is applied which is a separate network. Then a distance check filter is applied.
