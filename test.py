import pdb
import rl_tuner_ops
import sys

def load_dictionary(dict_file_path):
    char_to_int={}
    int_to_char={}
    with open(dict_file_path,'r') as f:
      for line in f:
        tmp=line.strip('\r\n').split(':')
        tmp[0]=tmp[0].strip()
        char_to_int[tmp[0]]=int(tmp[1])
        int_to_char[int(tmp[1])]=tmp[0]
    return char_to_int,int_to_char
def get_autocorrelation(sentence, lag=1):
    return rl_tuner_ops.autocorrelate(sentence, lag=lag)

# get number of repeat pairs in one sequence
## a b c c d d # contains 2 repeat pairs

def get_repeat(sentence):
    num_of_repeat=0
    for i in xrange(len(sentence)-1):
        if(sentence[i]==sentence[i+1]):
            num_of_repeat+=1
    return num_of_repeat

def load_tuples(filename, dicname):
    dic = {}
    tuples={}
    with open(dicname, 'r') as f:
      for line in f:
        strs = line.split(':')
        dic[strs[0].strip()] = int(strs[1].strip())
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        line = line.lower()
        strs = line.split()
        for i in range(len(strs) - 3):
          tuples[(dic[strs[i]], dic[strs[i+1]], dic[strs[i+2]])] = 1
    return tuples

def get_bleu(sentence,tuples):

    bleu = 0.0
    num_tri = 0.0
    num_exist = 0.0
    for i in range(len(sentence) - 3):
      num_tri += 1
      #pdb.set_trace()
      if(tuples.has_key((sentence[i], sentence[i+1], sentence[i+2]))):
        num_exist += 1
    
    if(num_tri > 0):
      bleu = num_exist / num_tri

    return bleu


tuples= load_tuples('ptb.train.txt', 'ptb_dict.txt')
char_to_int,int_to_char=load_dictionary('./ptb_dict.txt')
filename= sys.argv[1]

autocorrelation_lag1=0.0
autocorrelation_lag2=0.0
autocorrelation_lag3=0.0
num_of_sentence=0
num_of_words=0
num_of_repeat=0.0
bleu=0.0
with open(filename,'r') as f:
    for line in f:
        line=line.strip().split()
        if(len(line)==0): continue
        num_of_sentence+=1
        sentence=[]
        for w in line:
            sentence.append(char_to_int[w])
        num_of_words+=len(sentence)
        autocorrelation_lag1+=get_autocorrelation(sentence, lag=1)
        autocorrelation_lag2+=get_autocorrelation(sentence, lag=2)
        autocorrelation_lag3+=get_autocorrelation(sentence, lag=3)
        num_of_repeat+=get_repeat(sentence)
        bleu+=get_bleu(sentence,tuples)

autocorrelation_lag1=autocorrelation_lag1/num_of_sentence
autocorrelation_lag2=autocorrelation_lag2/num_of_sentence
autocorrelation_lag3=autocorrelation_lag3/num_of_sentence
bleu=bleu/num_of_sentence
print ("autocorrelation_lag1=",autocorrelation_lag1)
print ("autocorrelation_lag2=",autocorrelation_lag2)
print ("autocorrelation_lag3=",autocorrelation_lag3)
print ("repeat ratio",num_of_repeat/(num_of_words-num_of_sentence ))
print ("bleu=",bleu)

