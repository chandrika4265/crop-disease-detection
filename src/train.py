import tensorflow as tf

dataset_path = "PlantVillage"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

print("Classes Found:")
print(train_dataset.class_names)
import tensorflow as tf

dataset_path = "PlantVillage"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

print("Training Classes:")
print(train_dataset.class_names)

print("Training Batches:", len(train_dataset))
print("Validation Batches:", len(validation_dataset))
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)
model.save("Models/crop_disease_model.keras")

print("Model Saved Successfully!")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)
train = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
