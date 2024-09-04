import numpy as np
from sklearn.metrics import classification_report

# Load the predictions and labels
labels_path = './visualizations/Pretrained_labels.npy'
predictions_path = './visualizations/Pretrained_predictions.npy'

labels = np.load(labels_path)
predictions = np.load(predictions_path)

# Ensure the predictions and labels have the same length
assert len(labels) == len(predictions), "Predictions and labels must have the same length."

# Generate classification report
report = classification_report(labels, predictions, output_dict=True)

# Print the accuracy and predictions for each class
for class_id, metrics in report.items():
    if class_id.isdigit():  # Filter out 'accuracy', 'macro avg', 'weighted avg'
        print(f"Class {class_id}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")

# Optionally, print the overall accuracy
print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
