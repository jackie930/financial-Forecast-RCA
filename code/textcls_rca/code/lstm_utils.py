from torch import nn
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm


class BertClassifier_LSTM(nn.Module):

    def __init__(self, bert_model, dropout=0.1, bert_freeze=False):
        super(BertClassifier_LSTM, self).__init__()
        self.bert_freeze = bert_freeze
        self.bert = bert_model
        
        if self.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(768, 256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 2)
#         self.fc2 = nn.Linear(128, 2)
#         self.relu = nn.ReLU()

    def forward(self, input_id_0, input_id_1, input_id_2, mask_0, mask_1, mask_2):

        _, pooled_output_0 = self.bert(input_ids=input_id_0, attention_mask=mask_0,return_dict=False)
        _, pooled_output_1 = self.bert(input_ids=input_id_1, attention_mask=mask_1,return_dict=False)
        _, pooled_output_2 = self.bert(input_ids=input_id_2, attention_mask=mask_2,return_dict=False)
        pooled_output = torch.cat((pooled_output_0, pooled_output_1, pooled_output_2), axis=1).reshape((10,3,-1))
    
        _, (ht, ct) = self.lstm(pooled_output)
        x = ht.squeeze(0)
        x = self.dropout(x)
        y = self.fc1(x)
#         y = self.relu(self.fc2(x))
        
        return y


class Dataset_lstm(torch.utils.data.Dataset):
    def __init__(self, df, length, tokenizer):

        self.labels = [label for label in df['key_label']]
        self.tokens_1 = [tokenizer(text, 
                               padding='max_length', max_length = length, truncation=True,
                                return_tensors="pt") for text in df['input_1']]
        self.tokens_2 = [tokenizer(text, 
                               padding='max_length', max_length = length, truncation=True,
                                return_tensors="pt") for text in df['input_2']]
        self.tokens_3 = [tokenizer(text, 
                               padding='max_length', max_length = length, truncation=True,
                                return_tensors="pt") for text in df['input_3']]

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
    

    
def train_model(model, train_dataset, val_dataset, auroc, learning_rate, epochs):

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
    

    

def evaluate_model(model,  test_dataset, auroc):
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
    

