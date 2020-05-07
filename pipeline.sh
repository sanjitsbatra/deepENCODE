# Train classifier
python train.py ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1 4 4 16 0.25 4 1

# Impute tracks
python impute.py ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1/ ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1/model-04.hdf5 1 && mkdir ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1/results_model-04 && mv ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1/*.npy ../Results/run_batchnormFalse_CTTrue_ClassificationTrue_4_4_16_0.25_4_1/results_model-04/

# Score imputed tracks
python ../../../deepENCODE/score_classification.py ../../../Data/Testing_Data/ .
