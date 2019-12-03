
(1)
- 'all_data_samples.csv' contains all the groundtruth data required to train the classiffier.
- Each row is a data sample.
- The table has 11 columns: video_id, RCC, QTC and bounding box information of the two objects.

(2)
- 5-fold Cross validation is applied across all data given in 'all_data_samples.csv'.
- The data is split into 70% for training, 20% for validation and 10% for testing.
- The sample IDs of the 5 folds are provided in 'cv_training_sample_ids.csv', 'cv_validation_sample_ids.csv' and 'cv_testing_sample_ids.csv'. 

(3)
- 1-hold-out validation is applied across the video IDs.
- 
