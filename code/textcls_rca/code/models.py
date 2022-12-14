from torch import nn
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F

class TextClassifierModel(nn.Module):
    '''
    bert_name: 'bert-base-multilingual-cased'
    model_type: 'single' or 'cnn' or 'lstm'
    '''
    def __init__(self, bert_name='bert-base-multilingual-cased', bert_freeze=False, dropout=0.1):
        super(TextClassifierModel, self).__init__()      
        self.bert_freeze = bert_freeze
        self.bert = BertModel.from_pretrained(bert_name)
        
        if self.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.classifier_layers_single = nn.Sequential(nn.Linear(768, 2),
                                                       nn.ReLU(),
                                                       nn.Softmax(dim=1)
                                                       )
    
    def forward(self, input_id, mask):
        _, bert_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        y = self.classifier_layers_single(bert_output)
        return y 
                                                  


                                          
class TextClassifierModel_Sequential(nn.Module):
    '''
    bert_name: 'bert-base-multilingual-cased'
    model_type: 'cnn' or 'lstm'
    '''
    def __init__(self, bert_name='bert-base-multilingual-cased', model_type='lstm', bert_freeze=False, dropout=0.1):
        super(TextClassifierModel_Sequential, self).__init__()
        self.model_type = model_type                                           
        self.bert_freeze = bert_freeze
        self.bert = BertModel.from_pretrained(bert_name)
        
        if self.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False



        self.lstm = nn.LSTM(768, 256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1_lstm = nn.Linear(256, 2)
    
        self.conv1 = nn.Conv2d(1, 8, 2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 2, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(16*192, 768)
        self.fc2 = nn.Linear(768, 2)   
    
    def forward(self, input_id_0, input_id_1, input_id_2, mask_0, mask_1, mask_2):
        _, bert_output_0 = self.bert(input_ids=input_id_0, attention_mask=mask_0,return_dict=False)
        _, bert_output_1 = self.bert(input_ids=input_id_1, attention_mask=mask_1,return_dict=False)
        _, bert_output_2 = self.bert(input_ids=input_id_2, attention_mask=mask_2,return_dict=False)
        
        if self.model_type=='lstm':
            bert_output = torch.cat((bert_output_0, bert_output_1, bert_output_2), axis=1).reshape((10,3,-1))
            _, (ht, ct) = self.lstm(bert_output)
            x = ht.squeeze(0)
            x = self.dropout(x)
            y = self.fc1_lstm(x)
            
        elif self.model_type=='cnn':
            bert_output = torch.cat((bert_output_0, bert_output_1, bert_output_2), axis=1).reshape((10,1,3,-1)) 
            x = self.pool(F.relu(self.conv1(bert_output)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*1*192)
            x = self.dropout(self.fc1(x))
            y = self.fc2(x)

        return y

    
    
    
class TextClassifierModel_User(nn.Module):

    def __init__(self, bert_name='bert-base-multilingual-cased', dropout=0.1, bert_emb_size=768, user_emb_size=256, bert_freeze=False):
        super(TextClassifierModel_User, self).__init__()
        self.bert_freeze = bert_freeze
        self.bert_emb_size = bert_emb_size
        self.user_emb_size = user_emb_size

        self.bert = BertModel.from_pretrained(bert_name)
    
        if self.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            
        self.classification_layers = nn.Sequential(nn.Linear(self.bert_emb_size+self.user_emb_size, 2),
                                                   nn.ReLU(),
                                                   nn.Softmax(dim=1)
                                                  )

    def forward(self, input_id, mask, user_embs):
        _, text_embs = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        concated_embs = torch.cat((user_embs, text_embs), 1)
        predicted_labels = self.classification_layers(concated_embs)
        
        return predicted_labels
    


class Dataset_Sequential(torch.utils.data.Dataset):
    def __init__(self, df, max_length, tokenizer):
        self.tokenizer = tokenizer

        self.tokens_1 = [self.tokenizer(text, 
                               padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_1']]
        self.tokens_2 = [self.tokenizer(text, 
                               padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_2']]
        self.tokens_3 = [self.tokenizer(text, 
                               padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_3']]
        
        self.labels = [label for label in df['key_label']]
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.tokens_1[idx], self.tokens_2[idx], self.tokens_3[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y    


    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, max_length, tokenizer):
        
        self.tokenizer = tokenizer
        self.labels = [label for label in df['key_label']]
        self.texts = [self.tokenizer(text, padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_info']]       
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
            
    
                                                  
def train_model(model, train_dataset, val_dataset, auroc, learning_rate, epochs, batch_size):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    print('len(train_data_loader)', len(train_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ~~~~~~~ ', device)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr= learning_rate)

    train_acc_list = []
    train_auc_list = []
    val_acc_list = []
    val_auc_list = []
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            total_auc_train = 0
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)         
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                               
                auc_score_train = auroc(output.squeeze(-1), train_label)
                total_auc_train += auc_score_train
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0
            total_auc_val = 0
            with torch.no_grad():
                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)
            
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    auc_score_val = auroc(output.squeeze(-1), val_label)
                    total_auc_val += auc_score_val
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Train AUC: {total_auc_train/len(train_dataloader): .3f}\
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}\
                | Val AUC: {total_auc_val / len(val_dataloader): .3f}')
            train_acc_list.append(total_acc_train / len(train_dataset))
            train_auc_list.append(total_auc_train/len(train_dataloader))
            val_acc_list.append(total_acc_val / len(val_dataset))
            val_auc_list.append(total_auc_val/len(val_dataloader))
#     return train_acc_list, train_auc_list, val_acc_list, val_auc_list
        
    
def evaluate_model(model, test_dataset, auroc, batch_size):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('device ~~~~~~~ ', device)
    
    total_acc_test = 0
    total_auc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            
            auc_score_test = auroc(output, test_label)
            total_auc_test += auc_score_test     
            
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc


    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}\
             | Test AUC: {total_auc_test/len(test_dataloader): .3f}')
    


                                                
def train_model_sequential(model, train_dataset, val_dataset, auroc, learning_rate, epochs):

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10)
    print('len(train_data_loader)', len(train_dataloader))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device ~~~~~~~~~~ ', device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    
    train_acc_list = []
    train_auc_list = []
    val_acc_list = []
    val_auc_list = []
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            total_auc_train = 0
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                
                mask_0 = train_input[0]['attention_mask'].to(device)
                input_id_0 = train_input[0]['input_ids'].squeeze(1).to(device)
                
                mask_1 = train_input[1]['attention_mask'].to(device)
                input_id_1 = train_input[1]['input_ids'].squeeze(1).to(device)
                
                mask_2 = train_input[2]['attention_mask'].to(device)
                input_id_2 = train_input[2]['input_ids'].squeeze(1).to(device)
                
                output = model(input_id_0, input_id_1, input_id_2, mask_0, mask_1, mask_2)         
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                               
                auc_score_train = auroc(output.squeeze(-1), train_label)
                total_auc_train += auc_score_train
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0
            total_auc_val = 0
            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)

                    mask_0 = val_input[0]['attention_mask'].to(device)
                    input_id_0 = val_input[0]['input_ids'].squeeze(1).to(device)

                    mask_1 = val_input[1]['attention_mask'].to(device)
                    input_id_1 = val_input[1]['input_ids'].squeeze(1).to(device)

                    mask_2 = val_input[2]['attention_mask'].to(device)
                    input_id_2 = val_input[2]['input_ids'].squeeze(1).to(device)
                    
                    output = model(input_id_0, input_id_1, input_id_2, mask_0, mask_1, mask_2)       
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    auc_score_val = auroc(output.squeeze(-1), val_label)
                    total_auc_val += auc_score_val
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Train AUC: {total_auc_train/len(train_dataloader): .3f}\
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}\
                | Val AUC: {total_auc_val / len(val_dataloader): .3f}')
            train_acc_list.append(total_acc_train / len(train_dataset))
            train_auc_list.append(total_auc_train/len(train_dataloader))
            val_acc_list.append(total_acc_val / len(val_dataset))
            val_auc_list.append(total_auc_val/len(val_dataloader))
#     return train_acc_list, train_auc_list, val_acc_list, val_auc_list    
    

    

def evaluate_model_sequential(model,  test_dataset, auroc):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device ~~~~~~~ ', device)
    
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    total_auc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            
            mask_0 = test_input[0]['attention_mask'].to(device)
            input_id_0 = test_input[0]['input_ids'].squeeze(1).to(device)

            mask_1 = test_input[1]['attention_mask'].to(device)
            input_id_1 = test_input[1]['input_ids'].squeeze(1).to(device)

            mask_2 = test_input[2]['attention_mask'].to(device)
            input_id_2 = test_input[2]['input_ids'].squeeze(1).to(device)
            
            output = model(input_id_0, input_id_1, input_id_2, mask_0, mask_1, mask_2)  
            
            auc_score_test = auroc(output, test_label)
            total_auc_test += auc_score_test     
            
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc


    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}\
             | Test AUC: {total_auc_test/len(test_dataloader): .3f}')
    

class Dataset_with_user(torch.utils.data.Dataset):

    def __init__(self, df, max_length, tokenizer):

        self.labels = [label for label in df['key_label']]
        self.texts = [tokenizer(text, padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_info']]
        self.ids =[user_id for user_id in df['user_no']]
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    def get_batch_user_ids(self, idx):
        # Fetch a batch of user ids
        return self.ids[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_ids = self.get_batch_user_ids(idx)

        return batch_texts, batch_y, batch_ids
    
    
    
def train_model_user(model, train_dataset, val_dataset, auroc, user_embeddings, learning_rate, epochs, batch_size):
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    print('len(train_data_loader)', len(train_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ~~~~~~~ ', device)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr= learning_rate)

    train_acc_list = []
    train_auc_list = []
    val_acc_list = []
    val_auc_list = []
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            total_auc_train = 0
            for train_input, train_label, train_ids in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                train_user_embs = user_embeddings[train_ids].to(device)
                output = model(input_id, mask, train_user_embs)         
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                               
                auc_score_train = auroc(output.squeeze(-1), train_label)
                total_auc_train += auc_score_train
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0
            total_auc_val = 0
            with torch.no_grad():
                for val_input, val_label, val_ids in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    val_user_embs = user_embeddings[val_ids].to(device)
                    output = model(input_id, mask, val_user_embs)  
 
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    auc_score_val = auroc(output.squeeze(-1), val_label)
                    total_auc_val += auc_score_val
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Train AUC: {total_auc_train/len(train_dataloader): .3f}\
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f}\
                | Val AUC: {total_auc_val / len(val_dataloader): .3f}')
            train_acc_list.append(total_acc_train / len(train_dataset))
            train_auc_list.append(total_auc_train/len(train_dataloader))
            val_acc_list.append(total_acc_val / len(val_dataset))
            val_auc_list.append(total_auc_val/len(val_dataloader))
#     return train_acc_list, train_auc_list, val_acc_list, val_auc_list
        
    
    
def evaluate_model_user(model, test_dataset, auroc, user_embeddings, batch_size):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('device ~~~~~~~ ', device)
    
    total_acc_test = 0
    total_auc_test = 0
    with torch.no_grad():
        for test_input, test_label, test_ids in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            test_user_embs = user_embeddings[test_ids].to(device)
            output = model(input_id, mask, test_user_embs)  
            
            auc_score_test = auroc(output, test_label)
            total_auc_test += auc_score_test     
            
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc


    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}\
             | Test AUC: {total_auc_test/len(test_dataloader): .3f}')
    
