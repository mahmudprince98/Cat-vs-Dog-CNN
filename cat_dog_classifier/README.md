
# Cat vs Dog Classifier 🐱🐶
Simple image classifier for **Cats vs Dogs** with **both PyTorch and TensorFlow** implementations.  
Works with a basic **folder dataset** and includes **training + inference** scripts.

## 📦 Dataset
Use any dataset with this structure (Kaggle "Dogs vs Cats" works great):
```
data/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

> Tip: If you only have a single `train/` folder, you can create a quick split using tools like `split-folders` or by moving 10–20% of images to `val/`.

---

## 🚀 Quickstart (PyTorch)
```bash
pip install -r requirements.txt

# Train (simple CNN)
python train_torch.py --data_dir data --epochs 8 --batch_size 32 --img_size 160 --model simple

# Train (transfer learning with ResNet18)
python train_torch.py --data_dir data --epochs 6 --batch_size 32 --img_size 224 --model resnet18

# Inference
python infer_torch.py --weights runs_torch/best.pt --image_path sample.jpg --class_names "cat,dog"
```

---

## 🚀 Quickstart (TensorFlow/Keras)
```bash
pip install -r requirements.txt

# Train (simple CNN)
python train_tf.py --data_dir data --epochs 8 --batch_size 32 --img_size 160 --model simple

# Train (transfer with MobileNetV2)
python train_tf.py --data_dir data --epochs 6 --batch_size 32 --img_size 224 --model mobilenet

# Inference
python infer_tf.py --weights runs_tf/best.h5 --image_path sample.jpg --class_names "cat,dog"
```

---

## 🧰 Project Files
```
cat_dog_classifier/
  ├─ train_torch.py     # PyTorch training (simple CNN or resnet18)
  ├─ infer_torch.py     # PyTorch single-image inference
  ├─ train_tf.py        # TensorFlow/Keras training (simple CNN or MobileNetV2)
  ├─ infer_tf.py        # TensorFlow single-image inference
  ├─ requirements.txt
  └─ README.md
```

## 📝 Notes
- Both frameworks save the **best weights** into `runs_torch/best.pt` or `runs_tf/best.h5`.
- Use `--class_names "cat,dog"` to control label ordering for inference display.
- CPU works fine for the simple CNN variants; GPU recommended for transfer models.

## 📜 License
MIT — free to use on your GitHub portfolio.
