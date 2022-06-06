# NLP Disaster Prediction by Tweets - MODELING

from nlp_data_preprocessing import InitialDataLoader

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
import joblib
import sys
import nltk
import gc
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AdamW, BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, BertModel, RobertaModel, XLMRobertaModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


warnings.filterwarnings('ignore')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, pretrained = 'bert-base-uncased'):
        self.labels = df['target'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained) 
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.texts = [self.tokenizer(text, padding = 'max_length', max_length = 512, truncation = True,
                                return_tensors = 'pt') for text in df['new_text']]
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
    def __init__(self, dropout = 0.5, pretrained = 'bert-base-uncased'):
        super(BertClassifier, self).__init__()
        self.model = BertModel.from_pretrained(pretrained)
        #self.model = RobertaModel.from_pretrained('roberta-base')
        #self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.model(input_ids = input_id, attention_mask = mask, return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

class Model1(nn.Module):
    def __init__(self, dropout = 0.5, pretrained = 'bert-base-uncased'):
        super(Model1, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
            _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict = False)
            dropout_output = self.dropout(pooled_output)
            linear_output = self.linear(dropout_output)
            final_layer = self.relu(linear_output)
            return final_layer

class Model2(nn.Module):
    def __init__(self, dropout = 0.5, pretrained = 'bert-base-uncased'):
        super(Model2, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
    
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelA.fc = nn.Identity() # Remove last linear layer
        self.modelB.fc = nn.Identity()
        self.classifier = nn.Linear(1536, 2)
        
    def forward(self, x, mask):
        x1 = self.modelA(x.clone(), mask.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x.clone(), mask.clone())
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

# Clean GPU cache if necessary
gc.collect()
torch.cuda.empty_cache()

class Train():
    def __init__(self, model, train_data, val_data, criterion, optimizer, epochs, batch_size):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        
    def train_plots(self, epochs_list, train_losses, val_losses, train_accs, val_accs):
        #%matplotlib inline
        sns.set(rc={'figure.figsize':(19, 9)})
        fig, ax = plt.subplots(1,2)
        ax[0].plot(epochs_list, train_losses, label = 'Training Loss', marker='o')
        ax[0].plot(epochs_list, val_losses, label = 'Validation Loss', marker='o')
        ax[0].set_title('Loss Values')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Value')
        ax[1].plot(epochs_list, train_accs, label = 'Training Accuracy', marker='o')
        ax[1].plot(epochs_list, val_accs, label = 'Validation Accuracy', marker='o')
        ax[1].set_title('Accuracy Values')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Percent (%)')
        ax[0].legend()
        ax[1].legend()
        #plt.show()
        fig.savefig('./plots/train-val-loss-accs.png')

    def start_train(self):
        train, val = Dataset(self.train_data), Dataset(self.val_data)
        train_dataloader = torch.utils.data.DataLoader(train, self.batch_size, shuffle = True)
        val_dataloader = torch.utils.data.DataLoader(val, self.batch_size)

        use_cuda = torch.cuda.is_available()
        print('CUDA:', use_cuda)
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('You are using:', torch.cuda.get_device_name(device))
        total_steps = len(self.train_data)*self.epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)

        if use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        epochs_list = []
    
        for epoch_num in range(self.epochs):
            print('\n====================== Epoch {:} / {:} =====================\n'.format(epoch_num + 1, self.epochs))
            total_loss_train = 0
            total_acc_train = 0
            self.model.train()
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)

                batch_loss = self.criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc_tr = (output.argmax(dim = 1) == train_label).sum().item()
                total_acc_train += acc_tr
                
                self.model.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                    
                    acc_val = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc_val
            # Losses
            final_train_loss = total_loss_train/len(self.train_data)
            final_val_loss = total_loss_val/len(self.val_data)
            train_losses.append(final_train_loss)
            train_losses.sort(reverse=True)
            val_losses.append(final_val_loss)

            # Accuracies
            final_train_acc = (total_acc_train/len(self.train_data))*100
            final_val_acc = (total_acc_val/len(self.val_data))*100
            train_accs.append(final_train_acc)
            val_accs.append(final_val_acc)
            epochs_list.append(epoch_num + 1)
            
            # Plots
            self.train_plots(epochs_list, train_losses, val_losses, train_accs, val_accs)

            print(f'Train Loss: {final_train_loss: .3f} | Train Accuracy: {final_train_acc: .3f}%')
            print(f'Validation Loss: {final_val_loss: .3f} | Validation Accuracy: {final_val_acc: .3f}%\n')
            
            # Save the model
            torch.save(model.state_dict(), 'C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\models\\nlp_disaster_tweets_bert.pth')
            print('Model Has Been Saved!')

class Test():
    def __init__(self, model, test_data, batch_size):
        self.model = model
        self.test_data = test_data
        self.batch_size = batch_size
        
    def plot_metrics(self, labels, outputs):
        labels = torch.cat(labels, dim = 0)
        labels = labels.cpu().numpy()
        outputs = torch.cat(outputs, dim = 0)
        probs = F.softmax(outputs, dim = 1).cpu().numpy()
        preds = probs[:, 1]
        
        # ROC
        fpr, tpr, threshold = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        # Classification Report
        y_pred = np.where(preds > 0.5, 1, 0)
        
        print('\nClassification Report:\n', classification_report(labels, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(labels, y_pred)
        
        #%matplotlib inline
        sns.set(rc={'figure.figsize':(9, 6)})
        plt.gca().set_title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.gca().legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.gca().set_ylabel('True Positive Rate')
        plt.gca().set_xlabel('False Positive Rate')
        plt.savefig('./plots/roc-curve.png')
        
        #sns.set(rc={'figure.figsize':(7,5)})
        cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_disp.plot()
        print('Confusion Matrix:\n')
        print(cm)
        plt.show()

    def start_test(self):
        test = Dataset(self.test_data)
        test_dataloader = torch.utils.data.DataLoader(test, self.batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        if use_cuda:
            self.model = self.model.cuda()

        self.model.eval()
        total_acc_test = 0
        test_outputs = []
        test_labels = []
        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.type(torch.LongTensor)
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)
                acc = (output.argmax(dim = 1) == test_label).sum().item()
                total_acc_test += acc
                
                test_labels.append(test_label)
                test_outputs.append(output)
        self.plot_metrics(test_labels, test_outputs)
        print(f'Test Accuracy: {(total_acc_test / len(self.test_data))*100: .3f} %')

class UnseenDataset(torch.utils.data.Dataset):

    def __init__(self, df, pretrained = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.texts = [self.tokenizer(text, 
                               padding = 'max_length', max_length = 512, truncation = True,
                                return_tensors = 'pt') for text in df['new_text']]
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
        device = torch.device('cuda' if use_cuda else 'cpu')

        self.model.load_state_dict(torch.load(self.model_path, map_location = 'cpu'))

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


if __name__ == '__main__':

    """ #To launch ensemble 
    model1 = Model1()
    model2 = Model2()
    model = MyEnsemble(model1, model2)
    parameters = list(model1.parameters()) + list(model2.parameters())
    optimizer = AdamW(parameters, lr = 3e-6, eps = 1e-8) """

    model = BertClassifier()
    optimizer = AdamW(model.parameters(), lr = 3e-6, eps = 1e-8)
    loss_func = nn.CrossEntropyLoss()   
    epochs = 4
    batch_size = 5
    model_path = 'C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\models\\nlp_disaster_tweets_bert.pth'
    path = Path(model_path)
    initial_data_loader = InitialDataLoader('C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\train.csv', 'C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\test.csv')
    print('==================== Mode Selection ===================\n')
    user_input = input('- Press t to start training and testing\n- Press p to make predictions using the existing BERT model\n- Press q to exit\n')
    if (user_input == 't'):
        train_data_load = initial_data_loader.train_data_preprocessing()
        # I use 80% for train, 10% for validation and 10% for test
        train_data, val_data, test_data = np.split(train_data_load.sample(frac = 1, random_state = 42), [int(.8*len(train_data_load)), int(.9*len(train_data_load))])
        print('\n============= Train/Validation/Test Split =============\n')
        print('\nTrain/Validation/Test dataset size: \n', len(train_data), '/', len(val_data), '/', len(test_data))
        print('\n=================== Training Started ==================\n')
        train = Train(model, train_data, val_data, loss_func, optimizer, epochs, batch_size) 
        train.start_train()
        print('Training Completed!')
        print('\n==================== Testing Started ====================\n')
        test = Test(model, test_data, batch_size)
        test.start_test()
        print('Testing Completed!')
    else:
        if (user_input == 'p'):
            if path.is_file():
                test_data_load = initial_data_loader.test_data_preprocessing()
                print('============= Making Prediction =============')
                predict = Predict(model, model_path, test_data_load, batch_size)
                predict.start_predict()
                print('Predictions Made and Saved!')
            else:
                print('OOPS! THERE IS NO EXISTING BERT MODEL FOUND. PLEASE TRAIN AND TEST ONE IN ORDER TO HAVE ONE :)')
        else:
            try:
                raise ValueError('Please choose only between train/test, predict, or exit.')
                sys.exit()
            except ValueError as ve:
                print(ve)
        if (user_input == 'q'):
            sys.exit()

        

    
    