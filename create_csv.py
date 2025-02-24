import os
import pandas as pd


real_folder = 'UniversalFakeDetect/train/0_real'
fake_folder = 'UniversalFakeDetect/train/1_fake'

real_files = [f for f in os.listdir(real_folder) if os.path.isfile(os.path.join(real_folder, f))]
fake_files = [f for f in os.listdir(fake_folder) if os.path.isfile(os.path.join(fake_folder, f))]

real_data = [{'image': os.path.join(real_folder, f), 'text': 'real', 'label': 0} for f in real_files]
fake_data = [{'image': os.path.join(fake_folder, f), 'text': 'fake', 'label': 1} for f in fake_files]

combined_data = real_data + fake_data


df = pd.DataFrame(combined_data)
df.to_csv('trainFile.csv', index=False)

print("CSV file 'trainFile.csv' has been created.")
