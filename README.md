# Semi-supervised Fine-grained Approach for Arabic Dialect Detection
  - Team: TRY_NLP
  - NADI Shared task - WANLP Workshop Submission Notebooks
  - NADI website: https://sites.google.com/view/nadi-shared-task/home?authuser=0
  - Article: https://www.aclweb.org/anthology/2020.wanlp-1.25/
  
```
@inproceedings{appiah-balaji-b-2020-semi,
    title = "Semi-supervised Fine-grained Approach for {A}rabic dialect detection task",
    author = "Appiah Balaji, Nitin Nikamanth  and
      B, Bharathi",
    booktitle = "Proceedings of the Fifth Arabic Natural Language Processing Workshop",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wanlp-1.25",
    pages = "257--261",
    abstract = "Arabic being a language with numerous different dialects, it becomes extremely important to device a technique to distinguish each dialect efficiently. This paper focuses on the fine-grained country level and province level classification of Arabic dialects. The experiments in this paper are submissions done to the NADI 2020 shared Dialect detection task. Various text feature extraction techniques such as TF-IDF, AraVec, multilingual BERT and Fasttext embedding models are studied. We thereby, propose an approach of text embedding based model with macro average F1 score of 0.2232 for task1 and 0.0483 for task2, with the help of semi supervised learning approach.",
}
```

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
