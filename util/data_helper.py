from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def fruits_dataset(path):
    training_data = []
    training_label = []

    batches = ImageDataGenerator().flow_from_directory(directory=path, target_size=(64, 64),
                                                       batch_size=30)
    return batches



