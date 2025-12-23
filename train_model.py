import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import os

def train_flower_model():
    print("⬇️  Checking dataset...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    # --- 🛠️ THE FIX: Ensure we are pointing to the subfolders ---
    # If the current path doesn't have 'daisy', check if it's nested inside another folder
    if not (data_dir / 'daisy').exists():
        # Check if there is a 'flower_photos' folder inside
        potential_dir = data_dir / 'flower_photos'
        if (potential_dir / 'daisy').exists():
            data_dir = potential_dir
    
    print(f"📂 Data directory set to: {data_dir}")
    # ------------------------------------------------------------

    img_height = 180
    img_width = 180
    batch_size = 32

    print("⚙️  Preparing data...")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
            
    except ValueError as e:
        print("\n❌ ERROR: Could not find images.")
        print("Please delete the '.keras' folder in your user directory and try again.")
        return

    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}")
    
    # STOP if we still only see 'flower_photos'
    if len(class_names) == 1:
        print("⚠️  WARNING: Only 1 class found. Training will fail. Check data_dir path.")
        return

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build Model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(180, 180, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(180, 180, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    print("🚀 Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    model.save('flower_model.h5')
    
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print("✅ Fixed Model saved!")

if __name__ == "__main__":
    train_flower_model()