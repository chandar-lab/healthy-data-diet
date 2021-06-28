from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
from torch.optim import Adam
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import random
import time
import pandas as pd
from classifier import train_classifier
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax=torch.nn.Softmax(dim=1).to(device)


def training_epoch(epoch,args,optimizer,device,tokenizer,model,train_data,train_data_2):
    model.train()
    epoch_bias=torch.tensor(0.0).to(device)
    epoch_accuracy=torch.tensor(0.0).to(device)
    epoch_reward=torch.tensor(0.0).to(device)
    epoch_loss=torch.tensor(0.0).to(device)
  
    for i in range(int(np.floor(len(train_data)/args.batch_size))):
  
        torch.cuda.empty_cache()
        logs = dict()
  
        rewards_acc = []
        rewards_bias = []
        rewards_total = []
        accuracy = []
        
        #### get a batch from the dataset
        df_batch_1 = list(train_data["Tweets"].iloc[i*args.batch_size:(i+1)*args.batch_size])
        df_batch_1_tokenized = tokenizer(df_batch_1, padding=True, truncation=True, max_length=20)
  
        df_batch_2 = list(train_data_2["Tweets2"].iloc[i*args.batch_size:(i+1)*args.batch_size])
        df_batch_2_tokenized = tokenizer(df_batch_2, padding=True, truncation=True, max_length=20)
  
        results_1=model.forward(input_ids=torch.tensor(df_batch_1_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_1_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_1_tokenized['token_type_ids']).to(device))[0]        
        results_1=results_1.to(device)
  
        results_2=model.forward(input_ids=torch.tensor(df_batch_2_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_2_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_2_tokenized['token_type_ids']).to(device))[0]
        results_2=results_2.to(device)
  
        reward_bias=-torch.norm(results_1-results_2, dim=1)
        reward_acc=(torch.argmax(results_1, axis=1)==torch.tensor(train_data["Class"].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device)).double()
        rewards_bias.append(torch.tensor(reward_bias).to(device))
        rewards_acc.append(torch.tensor(reward_acc).to(device))
        rewards_total.append(torch.tensor(reward_bias+args.PG_lambda*reward_acc).to(device))
        accuracy.append(torch.sum(torch.argmax(results_1, axis=1)==torch.tensor(train_data["Class"].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device))/args.batch_size)
        print(accuracy)
        rewards = torch.cat(rewards_total).to(device)
  
        epoch_bias+= torch.mean(torch.stack(rewards_bias))
        epoch_accuracy+= torch.mean(torch.stack(accuracy))
        epoch_reward+= torch.mean(rewards)
  
        #### Run policy gradient training 
        t = time.time()
        loss=-torch.sum(torch.log(torch.max(softmax(results_1),axis=1)[0])*rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
    #### Log everything
    # logs.update(stats)
    logs['training_bias'] = -epoch_bias.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['training_reward_mean'] = epoch_reward.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['training_accuracy'] = epoch_accuracy.cpu().numpy()/(np.floor(len(train_data)/args.batch_size))
    logs['epoch'] = epoch
    print(logs['training_bias'], logs['training_reward_mean'] ,logs['training_accuracy'])
    wandb.log(logs)

def validation_epoch(epoch,args,device,tokenizer,model,validation_data,validation_data_2):
    model.eval()
    with torch.no_grad():
      epoch_bias=torch.tensor(0.0).to(device)
      epoch_accuracy=torch.tensor(0.0).to(device)
      epoch_reward=torch.tensor(0.0).to(device)
      epoch_loss=torch.tensor(0.0).to(device)
  
      for i in range(int(np.floor(len(validation_data)/args.batch_size))):
  
          logs = dict()
  
          rewards_acc = []
          rewards_bias = []
          rewards_total = []
          accuracy = []
          
          #### get a batch from the dataset
          df_batch_1 = list(validation_data["Tweets"].iloc[i*args.batch_size:(i+1)*args.batch_size])
          df_batch_1_tokenized = tokenizer(df_batch_1, padding=True, truncation=True, max_length=20)
  
          df_batch_2 = list(validation_data_2["Tweets2"].iloc[i*args.batch_size:(i+1)*args.batch_size])
          df_batch_2_tokenized = tokenizer(df_batch_2, padding=True, truncation=True, max_length=20)
  
          results_1=model.forward(input_ids=torch.tensor(df_batch_1_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_1_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_1_tokenized['token_type_ids']).to(device))[0]        
          results_1=results_1.to(device)
  
          results_2=model.forward(input_ids=torch.tensor(df_batch_2_tokenized['input_ids']).to(device),attention_mask=torch.tensor(df_batch_2_tokenized['attention_mask']).to(device),token_type_ids=torch.tensor(df_batch_2_tokenized['token_type_ids']).to(device))[0]
          results_2=results_2.to(device)
  
          reward_bias=-torch.norm(results_1-results_2, dim=1)
          reward_acc=(torch.argmax(results_1, axis=1)==torch.tensor(validation_data["Class"].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device)).double()
          rewards_bias.append(torch.tensor(reward_bias).to(device))
          rewards_acc.append(torch.tensor(reward_acc).to(device))
          rewards_total.append(torch.tensor(reward_bias+args.PG_lambda*reward_acc).to(device))
          accuracy.append(torch.sum(torch.argmax(results_1, axis=1)==torch.tensor(validation_data["Class"].iloc[i*args.batch_size:(i+1)*args.batch_size].tolist()).to(device))/args.batch_size)
          print(accuracy)
          rewards = torch.cat(rewards_total).to(device)
  
          epoch_bias+= torch.mean(torch.stack(rewards_bias))
          epoch_accuracy+= torch.mean(torch.stack(accuracy))
          epoch_reward+= torch.mean(rewards)
  
          #### Run PPO training 
          t = time.time()
          loss=-torch.sum(torch.log(torch.max(softmax(results_1),axis=1)[0])*rewards)
            
      #### Log everything
      # logs.update(stats)
      logs['validation_bias'] = -epoch_bias.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['validation_reward_mean'] = epoch_reward.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['validation_accuracy'] = epoch_accuracy.cpu().numpy()/(np.floor(len(validation_data)/args.batch_size))
      logs['epoch'] = epoch
      print(logs['validation_bias'], logs['validation_reward_mean'] ,logs['validation_accuracy'])
      wandb.log(logs)  


def run_experiment(args):

    # Define pretrained tokenizer and model
    
    wandb.init(name='run_1', project='debiasing_sexism_detection_twitter_PG', config=args)
    model,tokenizer=train_classifier(args)
    model=model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    train_data = pd.read_csv('./data/'+args.dataset+'_train_original_gender.csv')
    train_data_2 = pd.read_csv('./data/'+args.dataset+'_train_reversed_gender.csv')
    validation_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    validation_data_2 = pd.read_csv('./data/'+args.dataset+'_valid_reversed_gender.csv')
    
    output_dir = "./saved_models/"
    best_reward=-9999
     
    
    for epoch in range(args.num_epochs):
      training_epoch(epoch,args,optimizer,device,tokenizer,model,train_data,train_data_2)
      validation_epoch(epoch,args,device,tokenizer,model,validation_data,validation_data_2)

    return model,tokenizer
