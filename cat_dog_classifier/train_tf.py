
import os, argparse, math
import tensorflow as tf

def build_datasets(data_dir, img_size=160, batch_size=32):
    train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )
    val = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )
    class_names = train.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    norm = tf.keras.layers.Rescaling(1./255)
    train = train.shuffle(1000).map(lambda x,y: (norm(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.map(lambda x,y: (norm(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train, val, class_names

def simple_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def mobilenet_transfer(input_shape, num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # fine-tune later if needed
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--model", type=str, default="simple", choices=["simple","mobilenet"])
    ap.add_argument("--out_dir", type=str, default="runs_tf")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, val, class_names = build_datasets(args.data_dir, args.img_size, args.batch_size)
    num_classes = len(class_names)
    input_shape = (args.img_size, args.img_size, 3)

    if args.model == "simple":
        model = simple_cnn(input_shape, num_classes)
        lr = 1e-3
    else:
        model = mobilenet_transfer(input_shape, num_classes)
        lr = 1e-4

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, "best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False
    )

    print(model.summary())
    hist = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[ckpt])
    print("[DONE] Best model saved to runs_tf/best.h5")

if __name__ == "__main__":
    main()
