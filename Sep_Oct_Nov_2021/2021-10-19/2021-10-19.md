# Combination of undersampling with Edited Nearest Neighbors and oversampling with SMOTE

Before SMOTE for all category for Megalodon result(Megalodon v2.3.4 + Guppy 5.0.14): Counter({0: 5894606, 1: 9301876, 2: 1662})

The result in 2021/10/13 (rf_params_smote_gridsearch.txt) shows that SMOTE + Random forest with grid search can only achieve F1_macro 0.65407436099396 in training dataset and 0.653863642702075 in testing dataset with the best Parameters:  {'rfr__bootstrap': False, 'rfr__max_depth': 25, 'rfr__max_features': 'auto', 'rfr__min_samples_leaf': 2, 'rfr__min_samples_split': 2, 'rfr__n_estimators': 120}  
Based on the confusion_matrix, the major error came from the 5hmC class, i.e., the oversampling didn't learn much from the 5hmC class. So it's necessary to combine downsamping for 5mC/5C to "zoom out" the 5hmC features.

Tutorial: https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/


3. Use different strategies for Megalodon results

Tutorial: https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/

Imbalanced classsification issue: Skewed Class Distribution

Baggign and random forest are not suited to classification problems with a skewed class distribution

2.1 Easy Ensemble

Process: Creating balanced samples of the training dataset by selecting all examples from the minority class and a subset from the majority class.
