# NLP Disaster Prediction by Tweets - MODELING

from nlp_data_preprocessing import InitialDataLoader

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
from pathlib import Path


warnings.filterwarnings('ignore')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = df['target'].to_list()
        self.texts = [bert_tokenizer(text, padding = 'max_length', max_length = 512, truncation = True,
                                return_tensors="pt") for text in df['new_text']]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout = 0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask = mask, return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

# Clean GPU cache if necessary
gc.collect()
torch.cuda.empty_cache()

initial_data_loader = InitialDataLoader('C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\train.csv', 'C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\test.csv')
data = initial_data_loader.data_preprocessing()

# I use 80% for train, 10% for validation and 10% for test
np.random.seed(112)
train_data, val_data, test_data = np.split(data[0].sample(frac = 1, random_state = 42), [int(.8*len(data[0])), int(.9*len(data[0]))])

print('Train dataset size: ', len(train_data))
print('Validation dataset size: ', len(val_data))
print('Test dataset size: ', len(test_data))

class Train():
    def __init__(self, model, train_data, val_data, criterion, optimizer, epochs, batch_size):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def start_train(self):
        train, val = Dataset(self.train_data), Dataset(self.val_data)
        train_dataloader = torch.utils.data.DataLoader(train, self.batch_size, shuffle = True)
        val_dataloader = torch.utils.data.DataLoader(val, self.batch_size)

        use_cuda = torch.cuda.is_available()
        print("CUDA:", use_cuda)
        device = torch.device("cuda" if use_cuda else "cpu")
        print("You are using:", torch.cuda.get_device_name(device))
    
        #training_stats = []
        #train_losses = []
        total_steps = len(self.train_data)*self.epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(self.optimizer,  num_warmup_steps = 0,
                                                    num_training_steps = total_steps)

        if use_cuda:
                self.model = self.model.cuda()
                self.criterion = self.criterion.cuda()
    
        for epoch_num in range(self.epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_num + 1, self.epochs))
            total_loss_train = 0
            total_acc_tr = 0
            self.model.train()
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)
                #label_ids_tr = train_label.to('cpu').numpy()

                batch_loss = self.criterion(output, train_label)
                total_loss_train += batch_loss.item()

                #output = output.detach().cpu().numpy()
                #pred_flat_tr = np.argmax(output, axis=1).flatten()
                #labels_flat_tr = label_ids_tr.flatten()
                #acc_tr = np.sum(pred_flat_tr == labels_flat_tr)/len(labels_flat_tr)
                acc_tr = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_tr += acc_tr
                
                self.model.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                self.optimizer.step()
                scheduler.step()
            ############ Validation ###############
            total_acc_val = 0
            total_loss_val = 0
            self.model.eval()
            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.type(torch.LongTensor)
                    val_label = val_label.to(device)
                        
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = self.model(input_id, mask)
                    #label_ids = val_label.to('cpu').numpy()
                    
                    batch_loss = self.criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    #output = output.detach().cpu().numpy()
                    #pred_flat = np.argmax(output, axis=1).flatten()
                    #labels_flat = label_ids.flatten()
                    #acc = np.sum(pred_flat == labels_flat)/len(labels_flat)
                    acc_val = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc_val
                
            final_train_loss = total_loss_train/len(self.train_data)
            final_val_loss = total_loss_val/len(self.val_data)
                    
            #train_losses = train_losses.append(final_train_loss)
            #best_loss = min(train_losses)
            #print(train_losses)
            print(f'Epochs: {epoch_num + 1} | Train Loss: {final_train_loss: .3f} | Train Accuracy: {(total_acc_tr/len(self.train_data))*100: .3f} % \
             | Validation Loss: {final_val_loss: .3f} | Validation Accuracy: {(total_acc_val/len(self.val_data))*100: .3f} %')
            # Save the model
            #if (final_train_loss <= best_loss):
            #print('Saving current classifier...')
            torch.save(model.state_dict(), "C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\models\\nlp_disaster_tweets_bert.pth")
            print('Model Has Been Saved!')

class Test():
    def __init__(self, model, test_data, batch_size):
        self.model = model
        self.test_data = test_data
        self.batch_size = batch_size

    def start_test(self):
        test = Dataset(self.test_data)
        test_dataloader = torch.utils.data.DataLoader(test, self.batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:
                
                test_label = test_label.type(torch.LongTensor)
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)
                #label_ids = test_label.to('cpu').numpy()

                #output = output.detach().cpu().numpy()
                #pred_flat = np.argmax(output, axis=1).flatten()
                #labels_flat = label_ids.flatten()
                #acc = np.sum(pred_flat == labels_flat)/len(labels_flat)
                #total_acc_test += acc

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {(total_acc_test / len(self.test_data))*100: .3f} %')

class UnseenDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.texts = [bert_tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['new_text']]
    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)

        return batch_texts

class Predict():
    def __init__(self, model, model_path, unseen_data, batch_size):
        self.model = model
        self.model_path = model_path 
        self.unseen_data = unseen_data
        self.batch_size = batch_size

    def start_predict(self):
        test = UnseenDataset(self.unseen_data)
        test_dataloader = torch.utils.data.DataLoader(test, self.batch_size, shuffle = False)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))

        if use_cuda:
            self.model = self.model.cuda()
            
        predictions = []
        model.eval()
        with torch.no_grad():
            for test_input in test_dataloader:
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                predictions.append(output.cpu().numpy())
        predictions = np.concatenate(predictions, axis = 0)
        self.unseen_data['target'] = predictions.argmax(axis = 1)
        self.unseen_data['target'] = self.unseen_data['target'].astype(int)
        predicted_data = self.unseen_data[['id', 'target']]
        predicted_data.to_csv('submission-bert.csv', index = False)

model = BertClassifier()
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = 5e-5, eps = 1e-8)
epochs = 1
batch_size = 5
model_path = "C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\models\\nlp_disaster_tweets_bert.pth"
unseen_data = data[1]

if __name__ == '__main__':
    path = Path(model_path)
    if path.is_file():
        print('Predicting...')
        predict = Predict(model, model_path, unseen_data, batch_size)
        predict.start_predict()
        print('Predictions Made and Saved!')
    else:
        print('Training Started...')
        train = Train(model, train_data, val_data, loss_func, optimizer, epochs, batch_size)
        train.start_train()
        print('Training Completed!')
        print('Testing Started...')
        test = Test(model, test_data, batch_size)
        test.start_test()
        print('Testing Completed!')

    
    