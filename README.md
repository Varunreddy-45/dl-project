# Garbage Classification for Smart Waste Management using CNN

A simple, single-file Deep Learning project using a Convolutional Neural Network (CNN) to classify garbage and route items to smart municipal bins.

## 🚀 How to Run

Run the single program file:
```bash
python main.py
```

### What `main.py` Does:
1. **Dataset**: Automatically sets up the image dataset in `dataset/`.
2. **CNN Model**: Builds a 2-block Convolutional Neural Network.
3. **Training**: Trains for 4 epochs and prints train/test accuracy.
4. **Accuracy Graph**: Saves the accuracy plot as `accuracy_graph.png`.
5. **Smart Bin Prediction**: Predicts sample garbage items (Plastic, Glass, Paper, Metal, Cardboard, Trash) and assigns them to the appropriate smart bin (Blue, Green, Yellow, Orange, Grey).

### 🗑️ Smart Bins:
- **Cardboard / Paper**: 🔵 Blue Bin (Recyclable Dry)
- **Glass**: 🟢 Green Bin (Recyclable Glass)
- **Metal**: 🟡 Yellow Bin (Recyclable Metal)
- **Plastic**: 🟠 Orange Bin (Recyclable Plastic)
- **Trash**: ⚫ Grey Bin (General Non-Recyclable)
