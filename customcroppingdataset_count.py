from CustomDataset import CustomDataset
def calculate_class_counts(dataset):
   # Initialize dictionaries to hold the counts
   class_counts = {class_id: {'images': 0, 'labels': 0} for class_id in range(1, 4)}

   # Calculate the counts for the subset
   for image_id in dataset.image_ids:
       # Get the image info
       info = dataset.image_info[image_id]
       # Increment the image count for the class
       class_counts[info['num_ids'][0]]['images'] += 1
       # Increment the label count for the class
       class_counts[info['num_ids'][0]]['labels'] += len(info['polygons'])

   return class_counts

# Create the dataset objects
train_dataset = CustomDataset()
val_dataset = CustomDataset()
test_dataset = CustomDataset()

# Load the datasets and prepare them
train_dataset.load_custom('/home/e814/Documents/dataset-png', 'train')
train_dataset.prepare()
val_dataset.load_custom('/home/e814/Documents/dataset-png', 'val')
val_dataset.prepare()
test_dataset.load_custom('/home/e814/Documents/dataset-png', 'test')
test_dataset.prepare()

# Calculate the class counts for each subset
train_counts = calculate_class_counts(train_dataset)
val_counts = calculate_class_counts(val_dataset)
test_counts = calculate_class_counts(test_dataset)

# Print the counts
print('Train counts:', train_counts)
print('Validation counts:', val_counts)
print('Test counts:', test_counts)
