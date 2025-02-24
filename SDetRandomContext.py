import random
import transformers
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import random
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers.generation import GenerationConfig
import re

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(123)

device = "cuda" if torch.cuda.is_available() else "cpu"


image_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms=None, model=None):
        """
        Args:
            input_filename (str): Path to the CSV file containing image paths and labels.
            transforms (callable, optional): Optional transform to be applied on an image.
            model (torch.nn.Module, optional): Pre-trained model for extracting image embeddings.
        """
        self.data = pd.read_csv(input_filename)
        self.transforms = transforms
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_path = self.data.iloc[idx, 0]  
        label = self.data.iloc[idx, 2]      

        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transforms:
            image = self.transforms(image).to(device)

        # If a model is provided, extract image embeddings
        if self.model:
            image_embedding = self.get_image_embedding_and_label(image)
            return image_embedding, label, image_path
        else:
            return image, label, image_path

def process_csv(df, k):
    
    real_items = df[df[1] == "real"].values.tolist()
    fake_items = df[df[1] == "fake"].values.tolist()
    
    if not real_items or not fake_items:
        raise ValueError("Both real and fake classes must be present in the CSV.")
    
    selected = [random.choice(real_items), random.choice(fake_items)]
    remaining = df[~df.index.isin([df.index[df[0] == item[0]][0] for item in selected])].values.tolist()
    
    selected.extend(random.sample(remaining, min(k - 2, len(remaining))))
    
    batch_precontext = []
    precontext = [
        {'text': 'Is this photo real? Please provide your Answer. You should ONLY output "real" or "fake".'}
    ]
    
    for row in selected:
        precontext.append({'image': 'UniversalFakeDetect/train/' + row[0]})
        precontext.append({'text': 'User: It is \nAssistant: ' + row[1] + '\n'})
    
    batch_precontext.append(precontext)
    
    return batch_precontext

def extract_and_convert_label(text):
    
    if 'real' in text:
        return int(0)
    elif 'fake' in text:
        return int(1)
    
    match = re.search(r'\b(0|1)\b', text)
    if match:
        return int(match.group(0))


tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eod_id
modelRAG = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
modelRAG.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
processor = None


# Define the list of file names (test sets)
file_names = ['seeingdark.csv', 'san.csv', 'crn.csv','progan.csv', 'cyclegan.csv', 'biggan.csv', 'stylegan.csv', 'gaugan.csv', 'stargan.csv', 
              'deepfake.csv',  'imle.csv', 'guided.csv', 'ldm_200.csv', 'ldm_200_cfg.csv', 'ldm_100.csv', 'glide_100_27.csv', 'glide_50_27.csv', 
              'glide_100_10.csv', 'dalle.csv']

total_accuracy= 0

context_df = pd.read_csv('UniversalFakeDetect/train/trainFile.csv')

for file_name in file_names:
    all_test_labels = []
    all_test_preds = []

    datasetTest = CsvDataset('UniversalFakeDetect/test/' + file_name, image_transforms)
    test_dataloader = DataLoader(datasetTest, batch_size=2, shuffle=False)

    for _, Tlabels, image_paths in tqdm(test_dataloader):

        # Randomly Pick K element from train set as Context
        batch_precontext = process_csv(context_df, 3)

        # Add corresponding test images to the batch context
        for i in range(len(image_paths)):
            batch_precontext[i].append({'image': image_paths[i]})
            batch_precontext[i].append({'text': 'User: It is \nAssistant:'})

        # Tokenize batch
        total_inputs = [tokenizer.from_list_format(context) for context in batch_precontext]
        inputs = tokenizer(total_inputs, return_tensors='pt', padding=True, truncation=True).to(modelRAG.device)

        with torch.no_grad():
            preds = modelRAG.generate(**inputs, do_sample=False, max_new_tokens=3, min_new_tokens=1)

        input_token_len = inputs['input_ids'].shape[1]

        predicted_answers = tokenizer.batch_decode(preds[:, input_token_len:], skip_special_tokens=True)

        all_test_preds.extend([extract_and_convert_label(answer.lower()) for answer in predicted_answers])
        all_test_labels.extend(Tlabels.cpu().tolist()) 

    # Compute Accuracy
    # print(all_test_preds)
    accuracy = accuracy_score(all_test_labels, all_test_preds) * 100
    total_accuracy += accuracy
    print(f"Test Accuracy {file_name.split('.')[0]}: {accuracy:.2f}%")


print(f"Total Accuracy UniversalFakeDataset: {total_accuracy:.2f}%")
