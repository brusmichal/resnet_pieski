import numpy as np
import time
import os
import pandas as pd

from keras.applications.resnet import decode_predictions, ResNet50
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image, display
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

model = ResNet50(weights='imagenet')

test_filenames = os.listdir("test")
test_df = pd.DataFrame({
    'filename': test_filenames
})

images = ImageDataGenerator().flow_from_dataframe(
    test_df,
    "test",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(224, 224),
    batch_size=64
)

begin = time.monotonic()
predictions = model.predict(images, workers=16)
end = time.monotonic()

print('Duration for', len(predictions), 'images: %.3f sec' % (end - begin))
print('FPS : %.3f' % (len(predictions) / (end - begin)))

most_likely_labels = decode_predictions(predictions, top=3)

for i, img_path in enumerate(test_filenames[0:3]):
    display(Image('../input/dog-breed-identification/test/' + img_path))
    print(most_likely_labels[i])
