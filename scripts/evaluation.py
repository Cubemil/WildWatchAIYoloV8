from ultralytics import YOLO

# Load custom fine-tuned model
model = YOLO('./models/wildwatchyolov8_v2_finetuning05.pt')

# Validate the model on custom dataset
results = model.val(data='content/datasets/animalDataset')

# Print results
print(results)

# Access specific metrics
precision = results['metrics']['precision']
recall = results['metrics']['recall']
map50 = results['metrics']['map50']
map95 = results['metrics']['map']

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"mAP@0.50: {map50}")
print(f"mAP@0.50:0.95: {map95}")
