# sign_detection

## Dataset
1. Archived International Logistic Mark from Google IMG, Naver
2. Cropped individually
3. Preprocessed (200,200,3) - opencv

## Training
1. Machine Learning
    - Binary Classification
    - Random Foreset
    - One of One (One os Rest)
    - LightGBM (lower auc comparing RF)
2. Image Classification
    - VGG16 (pretrained, imagenet)
    - Resnet50 (train failure)
3. Objecet Detection
    - Scaled YOLOv4
