import os
import sys
import numpy as np
from chrmt_generator import DataGenerator
from chrmt_generator import MASK_VALUE
from chrmt_train import custom_loss
from tensorflow.keras.models import load_model
from tqdm import tqdm


def create_masked_custom(x, positions):

    # dimensions are window_size x len(ASSAY_TYPES)
    # we mask out some portions by setting them to mask_value
    for i in range(x.shape[0]):
        if i in positions:
            print(i, positions)
            x[i, :] = MASK_VALUE

    return x


if __name__ == '__main__':

    trained_model_path = sys.argv[1]
    trained_model = load_model(trained_model_path,
                               custom_objects={'custom_loss': custom_loss})

    window_size = int(sys.argv[2])

    test_generator = DataGenerator(window_size,
                                   1,
                                   shuffle=False,
                                   mode='test',
                                   masking_probability=0.0)

    np.set_printoptions(precision=3, suppress=True)

    for i in tqdm(range(len(test_generator))):

        X, Y = test_generator.__getitem__(i)

        for p in range(window_size):
            positions = [p]
            X_copy = np.copy(X)
            X_masked = np.expand_dims(create_masked_custom(X_copy[0],
                                                           positions), axis=0)
            yPred = trained_model.predict(X_masked)
            print(i, p, X_masked.shape, yPred.shape, "\n", X, "\n",
                  X_masked, "\n", yPred, "\n\n")

        if(i == 5):
            os._exit(1)
