from pymilvus import (
    connections,
    Collection
)
import random
import transformers
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.metrics import accuracy_score
from transformers.generation import GenerationConfig
import re


def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

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

class C2P_CLIP(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14', num_classes=1):
        super(C2P_CLIP, self).__init__()
        self.model= CLIPModel.from_pretrained(name)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        
        self.model.vision_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.fc = nn.Linear( 768, num_classes )
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.model.config.output_hidden_states,
            return_dict          = self.model.config.use_return_dict,      
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    

    def forward(self, img):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)
device = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = torch.load('./C2P_CLIP_release_20240901.pth' , map_location = "cpu")
model = C2P_CLIP( name='openai/clip-vit-large-patch14', num_classes=1 )
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()
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
            return image, label

    def get_image_embedding_and_label(self, image_input):
        """
        Extracts image embeddings using the provided model.

        Args:
            image_input (torch.Tensor): The image tensor.

        Returns:
            image_embedding (numpy.ndarray): The normalized image embedding.
        """
        # Ensure the image tensor has a batch dimension
        if len(image_input.shape) == 3:
            image_input = image_input.unsqueeze(0)  # Add batch dimension

        # Extract features using the model
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        # Normalize the features
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        # Return the embedding as a numpy array
        return image_features.squeeze(0).cpu().numpy().astype(np.float32)
    
def majority_voting(classes):
    # Count occurrences of each class
    class_counts = Counter(classes)

    # Return the most common class (majority voting)
    predicted_class = class_counts.most_common(1)[0][0]
    # print(predicted_class)
    
    return predicted_class


def search_and_query_batch(query_vectors_batch, search_field, limit):
    collection = Collection("SDetRAG_C2PCLIP")
    collection.load()

    search_params = {
        "metric_type": "IP", 
        "params": {"nprobe": 128}
    }

    # Perform batch search
    results_batch = collection.search(
        data=query_vectors_batch,  
        anns_field=search_field,  
        param=search_params, 
        limit=limit,  
        output_fields=["pk", "img_path", "label"]  
    )

    batch_precontext = []
    
    for results in results_batch:
        precontext = [
            {'text': 'Is this photo real? Please provide your Answer. You should ONLY output "real" or "fake".'}
        ]
        
        for hit in results:
            precontext.append({'image': 'UniversalFakeDetect/train/' + hit.entity.get('img_path')})
            precontext.append({'text': 'User: It is \nAssistant: ' + hit.entity.get('label').split('_')[-1] + '\n'})
        
        batch_precontext.append(precontext)

    return batch_precontext


# def search_and_query(query_vectors, search_field, limit):

#     collection = Collection("SDetRAG_C2PCLIP")
#     collection.load()


#     search_params = {
#     "metric_type": "IP", 
#     "params": {"nprobe": 128}
#     }

#     # Perform the search
#     results = collection.search(
#         data=query_vectors, 
#         anns_field=search_field,  # Replace with your vector field name
#         param=search_params, 
#         limit=limit,  # Number of results to return
#         output_fields=["pk", "img_path","label"] 
#     )

#     # Print the results
#     precontext = [
#         {'text': 'Is this photo real? Please provide your Answer. You should ONLY output "real" or "fake".'}
#     ]
#     for hits in results:
#         # labelsTemp = []
#         for i, hit in enumerate(hits):
#             precontext.append({'image': 'UniversalFakeDetect/train/'+hit.entity.get('img_path')})
#             precontext.append({'text': 'User: It is \nAssistant: '+ hit.entity.get('label').split('_')[-1]+'\n'})

#     return precontext


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



connect_to_milvus()

# Define the list of file names (test sets)
file_names = ['seeingdark.csv', 'san.csv', 'crn.csv','progan.csv', 'cyclegan.csv', 'biggan.csv', 'stylegan.csv', 'gaugan.csv', 'stargan.csv', 
              'deepfake.csv',  'imle.csv', 'guided.csv', 'ldm_200.csv', 'ldm_200_cfg.csv', 'ldm_100.csv', 'glide_100_27.csv', 'glide_50_27.csv', 
              'glide_100_10.csv', 'dalle.csv']

total_accuracy= 0

for file_name in file_names:
    all_test_labels = []
    all_test_preds = []

    datasetTest = CsvDataset('UniversalFakeDetect/test/' + file_name, image_transforms, model)
    test_dataloader = DataLoader(datasetTest, batch_size=2, shuffle=False)

    for query_vectors, Tlabels, image_paths in tqdm(test_dataloader):
        batch_embeddings = [query.numpy() for query in query_vectors]

        # Perform batch search in Milvus
        batch_precontext = search_and_query_batch(batch_embeddings, 'embeddings', 3)

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


# for file_name in file_names:
#     all_test_labels = []
#     all_test_preds = []

#     datasetTest = CsvDataset('UniversalFakeDetect/test/' + file_name, image_transforms, model)
#     test_dataloader = DataLoader(datasetTest, batch_size=1, shuffle=False)

#     for query_vectors, Tlabel, image_path in tqdm(test_dataloader):

#         batch_embeddings = [query.numpy() for query in query_vectors]
#         precontext = search_and_query(batch_embeddings, 'embeddings', 3)

#         precontext.append({'image': image_path[0]})
#         precontext.append({'text': 'User: It is \nAssistant:'})

#         # print(precontext)
#         total_inputs = tokenizer.from_list_format(precontext)
#         inputs = tokenizer(total_inputs, return_tensors='pt')
#         inputs = inputs.to(modelRAG.device)

#         with torch.no_grad():
#             pred = modelRAG.generate(**inputs, do_sample=True, max_new_tokens=3, min_new_tokens=1)
#         input_token_len = inputs['input_ids'].shape[1]
#         predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)

#         # print('Prediction: ',extract_and_convert_label(predicted_answers.lower()))

#         all_test_preds.append(extract_and_convert_label(predicted_answers.lower()))
#         all_test_labels.append(Tlabel)


#     print(all_test_preds)
#     accuracy = accuracy_score(all_test_labels, all_test_preds) * 100
#     print(f"Test Accuracy {file_name.split('.')[0]}: {accuracy:.2f}%")

print(f"Total Accuracy UniversalFakeDataset: {total_accuracy:.2f}%")
connections.disconnect("default")