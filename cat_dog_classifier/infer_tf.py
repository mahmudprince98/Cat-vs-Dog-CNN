
import argparse, numpy as np, tensorflow as tf
from PIL import Image

def load_and_preprocess(image_path, img_size):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .h5 model")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--class_names", type=str, default="cat,dog")
    args = ap.parse_args()

    class_names = [s.strip() for s in args.class_names.split(",")]
    model = tf.keras.models.load_model(args.weights)
    x = load_and_preprocess(args.image_path, args.img_size)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    print(f"Prediction: {class_names[idx]}  (confidence={probs[idx]:.4f})")

if __name__ == "__main__":
    main()
