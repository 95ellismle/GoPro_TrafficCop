from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


filepath = Path("/Users/mattellis/Projects/GoProCam/runs/detect/train/results.csv")

df = pd.read_csv(filepath, sep='\s*,\s*', engine="python")

f, all_a = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

map_ = {'train': 'Training', 'val': 'Validation',
        'box_loss': 'Box Loss', 'cls_loss': 'Class Loss',
        'dfl_loss': 'Distribution Focal Loss'}
for coli, col in enumerate(('train', 'val')):
    for rowi, row in enumerate(('box_loss', 'cls_loss', 'dfl_loss')):
        a = all_a[coli][rowi]
        a.plot(df['epoch'], df[f'{col}/{row}'])
        a.set_title(f"{map_[col]}\n{map_[row]}")
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

plt.show()
