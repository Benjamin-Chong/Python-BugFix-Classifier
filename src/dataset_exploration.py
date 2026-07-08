from datasets import load_dataset
from collections import Counter

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("iberu/RunBugRun")

print(ds.keys())
print(ds['train'].column_names)
print(ds['train'][0])

#investigating: distribution of labels for top 20 and the number of unique labels
label_counter = Counter()

for row in ds['train']: #iterates through all of the examples in the training only, counting each label
    if row['labels']:
        label_counter[row['labels'][0]] += 1

print('\nTop 20 Labels:')
for label, count in label_counter.most_common(20):
    print(label, count)

print('\n Number of Unique Primary Labels:', len(label_counter))

#investigating: how many labels per example
label_count_dist = Counter()
missing_labels = 0

for row in ds['train']:
    if row['labels'] is None:
        missing_labels += 1
        continue
    label_count_dist[len(row["labels"])] += 1

print("Labels per example:")
for num_labels, count in sorted(label_count_dist.items()):
    print(f'{num_labels} labels: {count}')

#investigating: how many top level categories exist
unique_label_category = Counter()

for row in ds['train']:
    if row['labels'] == None:
            continue
    for label in row['labels']:
        category = label.split('.')[0]
        unique_label_category[category] += 1

for name, count in sorted(unique_label_category.items()):
    print(f'{name} - {count}')

#investigating: checking if the dataset is all python by looking at random examples
for example in [1, 20, 50, 100, 200, 4000, 20000]:
     code = ds['train'][example]['buggy_code']
     print(f'{example} - Code: {code}')

#investigating: the try-catch label. python has try and except
for row in ds['train']:
    if row['labels'] is None:
        continue
    for label in row['labels']:
        if label.split('.')[0] == 'try_catch':
            print(f"{label} - Code: {row['buggy_code']}")

#investigating: how many unique labels per example
unique_amount_label_counter = Counter()

for row in ds['train']:
    if row['labels'] is None:
        continue
    else:
        unique_label_in_row = set()
        for label in row['labels']:
            label = label.split('.')[0]
            unique_label_in_row.add(label)
            
        unique_amount_label_counter[len(unique_label_in_row)] += 1

for key, value in unique_amount_label_counter.most_common(5):
    print(f'Unique Identifier Amount: {key} - Frequency: {value}')