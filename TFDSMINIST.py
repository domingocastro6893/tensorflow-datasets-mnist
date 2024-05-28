import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
mnist_data = tfds.load("fashion_mnist")

# Inspect available splits
for item in mnist_data:
    print("Item "+ item)

# Load the training split
mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

# Inspect the first record
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])

# Load dataset with additional information
mnist_test, info = tfds.load(name="fashion_mnist", with_info=True)
print(info)
