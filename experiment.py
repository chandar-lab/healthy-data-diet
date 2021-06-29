from torch.optim import Adam
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from classifier import train_classifier
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax=torch.nn.Softmax(dim=1).to(device)


def training_epoch(epoch,args,optimizer,device,tokenizer,model,train_data,train_data_gender_swap):
    
    logs = dict()
    model.train()
    compute_gradient = True
    epoch_bias, epoch_accuracy, epoch_reward, loss = epoch_loss (epoch,args,optimizer,device,tokenizer,model,train_data,train_data_gender_swap,compute_gradient)     

    #### Log everything
    logs['training_bias'] = -epoch_bias.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['training_reward_mean'] = epoch_reward.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['training_accuracy'] = epoch_accuracy.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['epoch'] = epoch
    wandb.log(logs)

def validation_epoch(epoch,args,optimizer,device,tokenizer,model,validation_data,validation_data_gender_swap):

    logs = dict()
    model.eval()
    compute_gradient = False

    with torch.no_grad():
      epoch_bias, epoch_accuracy, epoch_reward, loss = epoch_loss (epoch,args,optimizer,device,tokenizer,model,validation_data,validation_data_gender_swap,compute_gradient)     
      #### Log everything
      logs['validation_bias'] = -epoch_bias.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['validation_reward_mean'] = epoch_reward.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['validation_accuracy'] = epoch_accuracy.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['epoch'] = epoch
      wandb.log(logs)  

def epoch_loss(epoch,args,optimizer,device,tokenizer,model,data,data_gender_swap,compute_gradient):

      epoch_bias=torch.tensor(0.0).to(device)
      epoch_accuracy=torch.tensor(0.0).to(device)
      epoch_reward=torch.tensor(0.0).to(device)
      epoch_loss=torch.tensor(0.0).to(device)

      input_column_name = data.columns[1]
      input_column_name_gender_swap = data_gender_swap.columns[1]
      label_column_name = data.columns[2]
  
      for i in range(int(np.ceil(len(data)/args.batch_size))):
  
          rewards_acc, rewards_bias, rewards_total, accuracy = [] , [] , [] , []

          # the actual batch size is the same as args.batch_size unless it is the last batch becuase it will be smaller than that.
          if(i==int(np.floor(len(data)/args.batch_size))):
            actual_batch_size = len(data) % args.batch_size
          else:
            actual_batch_size = args.batch_size
                    
          #### get a batch from the dataset
          df_batch_original_gender = list(data[input_column_name].iloc[i*args.batch_size:(i+1)*args.batch_size])
          df_batch_original_gender_tokenized = tokenizer(df_batch_original_gender, padding=True, truncation=True, max_length=args.max_length)
  
          df_batch_gender_swap = list(data_gender_swap[input_column_name_gender_swap].iloc[i*args.batch_size:(i+1)*args.batch_size])
          df_batch_gender_swap_tokenized = tokenizer(df_batch_gender_swap, padding=True, truncation=True, max_length=args.max_length)
  
          results_original_gender=model.forward(input_ids=torch.tensor(df_batch_original_gender_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_original_gender_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_original_gender_tokenized['token_type_ids']).to(device))[0]        
          results_gender_swap=model.forward(input_ids=torch.tensor(df_batch_gender_swap_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_gender_swap_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_gender_swap_tokenized['token_type_ids']).to(device))[0]
  
          reward_bias=-torch.norm(results_original_gender-results_gender_swap, dim=1).to(device)
          reward_acc=(torch.argmax(results_original_gender, axis=1)==torch.tensor(data[label_column_name].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device)).double()
          
          rewards_bias.append(torch.tensor(reward_bias))
          rewards_acc.append(torch.tensor(reward_acc))
          rewards_total.append(torch.tensor(reward_bias+args.PG_lambda*reward_acc))

          accuracy.append(torch.sum(torch.argmax(results_original_gender, axis=1)==torch.tensor(data[label_column_name].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device))/actual_batch_size)
          print(accuracy)
          rewards = torch.cat(rewards_total)
  
          epoch_bias+= torch.mean(torch.stack(rewards_bias))
          epoch_accuracy+= torch.mean(torch.stack(accuracy))
          epoch_reward+= torch.mean(rewards)
  
          #### Run the policy gradient algorithm
          loss=-torch.sum(torch.log(torch.max(softmax(results_original_gender),axis=1)[0])*rewards)
          if(compute_gradient == True):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      return epoch_bias, epoch_accuracy, epoch_reward, loss

def run_experiment(args):

    # Define pretrained tokenizer and model
    
    wandb.init(name="lambda = " + str(args.PG_lambda), project='debiasing_sexism_detection_twitter_PG', config=args)
    model,tokenizer=train_classifier(args)
    model=model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    train_data = pd.read_csv('./data/'+args.dataset+'_train_original_gender.csv')
    train_data_gender_swap = pd.read_csv('./data/'+args.dataset+'_train_gender_swap.csv')
    validation_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    validation_data_gender_swap = pd.read_csv('./data/'+args.dataset+'_valid_gender_swap.csv')
        
    for epoch in range(args.num_epochs):
      training_epoch(epoch,args,optimizer,device,tokenizer,model,train_data,train_data_gender_swap)
      validation_epoch(epoch,args,optimizer,device,tokenizer,model,validation_data,validation_data_gender_swap)

    return model,tokenizer
