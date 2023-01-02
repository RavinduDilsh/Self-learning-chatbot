import seq2seq_wrapper
import importlib
from datetime import datetime
from datetime import date
import calendar
import webbrowser
import pyjokes
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2


# Importing the dataset
metadata, idx_q, idx_a = data_preprocessing.load_data(PATH = './')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX, validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()


# Building the seq2seq model
model = seq2seq_wrapper.Seq2Seq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)


session = model.restore_last_session()

def respond(question):
    encoded_question = data_utils_2.encode(question, w2idx, limit['maxq'])
    answer = model.predict(session, encoded_question)[0]
    return data_utils_2.decode(answer, idx2w) 

def wishMe():
    hour = int(datetime.now().hour)
    if hour>=0 and hour<12:
        print('\nGood Morning!...\n')
    elif hour>=12 and hour<18:
        print('\nGood Afternoon!...\n')
    else:
        print('\nGood Evening!...\n')
        
        

if __name__=="__main__":
    wishMe()
    print('Hello my name is Sam. What should i call you\n')
    user_name = input()
    print('\nWelcome '+user_name)
    while True :
      question = input(user_name+" : ")
      if question == 'Goodbye' or question == 'bye':
          print('\nQuiting....')
          break
      elif 'time' in question:
          now = datetime.now()
          print(now.strftime("%H:%M:%S"))
      elif 'date' in question:
          now = datetime.now()
          print(now.strftime("%Y-%m-%d "))
      elif 'today' in question:
          today = date.today()
          print(calendar.day_name[today.weekday()])
      elif 'google' in question:
          url = "https://www.google.com/search?q="
          print('Sam: Tell me what you want to search on Google')
          query = input("You: ")
          print('Sam : OK searching...')
          webbrowser.open_new(url+query)
      elif 'youtube' in question:
          url = "https://www.youtube.com/search?q="
          print('Sam: Tell me what you want to search on Youtube')
          query = input("You: ")
          print('Sam : OK searching...')
          webbrowser.open_new(url+query)
      elif 'calculate' in question:
          print('Sam: Tell me what you want to calculate')
          calc = input('You: ')
          print('Sam: '+str(eval(calc))+' is your answer ')
      elif 'joke' in question:
          print("Sam: "+pyjokes.get_joke())
      else:
          answer = respond(question)
          print ("Sam: "+answer)
