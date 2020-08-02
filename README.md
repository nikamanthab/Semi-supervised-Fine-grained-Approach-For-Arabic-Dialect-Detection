# Semi-supervised Fine-grained Approach for Arabic Dialect Detection
  - Team: TRY_NLP
  - NADI Shared task - WANLP Workshop Submission Notebooks
  - NADI website: https://sites.google.com/view/nadi-shared-task/home?authuser=0

### Features Extracted
  - Various pretrained feature embeddings trained on Arabic corpus, such as AraVec, fastText, BERT was studied.
  - Pretrained fastText gave better performance.

### Semi-supervised Structure
  - Used a combination of labelled to generate pseudolabels for unlabelled 10M data set and the top confidence predictions were added to the labelled data set for retraining.
  - This process was repeated multiple times.



### Performance
The model was trained on 21,000 labelled and 10M unlabelled data set.

The test performance:
| Task    | macro F1 score | accuracy | 
| ------  | -------- | ------ |
| Task1 (21 classes)  | 0.2004     | 0.3366 |
| Task2 (100 classes) | 0.0403    | 0.0486 |
