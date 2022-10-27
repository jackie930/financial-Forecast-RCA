from torch import nn
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, max_length, tokenizer):

        self.labels = [label for label in df['key_label']]
        self.texts = [tokenizer(text, padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['input_info']]
        # what info as input: title_abs or title_abs_content
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
    
    

class BertClassifier(nn.Module):

    def __init__(self, bert_model, bert_freeze=False, dropout=0.1):

        super(BertClassifier, self).__init__()
        self.dropout = 0.1
        self.bert_freeze = bert_freeze
        
        # BERT: Texts --> Embedding (CLS)
        self.bert = bert_model 

        ## To freeze the BERT Parameters, 
        if self.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            
        # Classification: Embedding --> Probability of [0,1] labels (SoftMax)
        self.classification_layers = nn.Sequential(nn.Linear(768, 2),
#                                                    nn.Dropout(self.dropout),
                                                   nn.ReLU(),
                                                   nn.Softmax(dim=1)
                                                  )

    def forward(self, input_id, mask):
        _, bert_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        predicted_labels = self.classification_layers(bert_output)
        
        return predicted_labels
    

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
    return train_acc_list, train_auc_list, val_acc_list, val_auc_list
        
    
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
    
