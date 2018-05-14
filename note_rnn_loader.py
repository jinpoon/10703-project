# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a class and operations for the MelodyRNN model.

Note RNN Loader allows a basic melody prediction LSTM RNN model to be loaded
from a checkpoint file, primed, and used to predict next notes.

This class can be used as the q_network and target_q_network for the RLTuner
class.

The graph structure of this model is similar to basic_rnn, but more flexible.
It allows you to either train it with data from a queue, or just 'call' it to
produce the next action.

It also provides the ability to add the model's graph to an existing graph as a
subcomponent, and then load variables from a checkpoint file into only that
piece of the overall graph.

These functions are necessary for use with the RL Tuner class.
"""

import os,pdb
import random
import time
# internal imports

import numpy as np
import tensorflow as tf
'''
import magenta
from magenta.common import sequence_example_lib
from magenta.models.rl_tuner import rl_tuner_ops
from magenta.models.shared import events_rnn_graph
from magenta.music import melodies_lib
from magenta.music import midi_io
from magenta.music import sequences_lib
'''
from keras.utils import np_utils
import rl_tuner_ops
class NoteRNNLoader(object):
  """Builds graph for a Note RNN and instantiates weights from a checkpoint.

  Loads weights from a previously saved checkpoint file corresponding to a pre-
  trained basic_rnn model. Has functions that allow it to be primed with a MIDI
  melody, and allow it to be called to produce its predictions for the next
  note in a sequence.

  Used as part of the RLTuner class.
  """

  def __init__(self, graph, scope, checkpoint_dir, checkpoint_file=None,
               midi_primer=None, training_file_list=None, hparams=None,
               note_rnn_type='default', checkpoint_scope='rnn_model',is_training=False,keep_prob=0.5):
    """Initialize by building the graph and loading a previous checkpoint.

    Args:
      graph: A tensorflow graph where the MelodyRNN's graph will be added.
      scope: The tensorflow scope where this network will be saved.
      checkpoint_dir: Path to the directory where the checkpoint file is saved.
      checkpoint_file: Path to a checkpoint file to be used if none can be
        found in the checkpoint_dir
      midi_primer: Path to a single midi file that can be used to prime the
        model.
      training_file_list: List of paths to tfrecord files containing melody
        training data.
      hparams: A tf_lib.HParams object. Must match the hparams used to create
        the checkpoint file.
      note_rnn_type: If 'default', will use the basic LSTM described in the
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      checkpoint_scope: The scope in lstm which the model was originally defined
        when it was first trained.
    """
    self.graph = graph
    self.session = None
    self.scope = scope
    self.batch_size = 1
    self.midi_primer = midi_primer
    self.checkpoint_scope = checkpoint_scope
    self.note_rnn_type = note_rnn_type
    self.training_file_list = training_file_list
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = checkpoint_file
    self.keep_prob=keep_prob
    self.is_training=is_training
    self.char_to_int={}
    self.int_to_char={}
    if hparams is not None:
      tf.logging.info('Using custom hparams')
      self.hparams = hparams
    else:
      tf.logging.info('Empty hparams string. Using defaults')
      self.hparams = rl_tuner_ops.default_hparams()

    self.build_graph()
    self.state_value = self.get_zero_state()
   # pdb.set_trace()
    if midi_primer is not None:
      self.load_primer()

    self.variable_names = rl_tuner_ops.get_variable_names(self.graph,
                                                          self.scope)

    self.transpose_amount = 0

  def get_zero_state(self):
    """Gets an initial state of zeros of the appropriate size.

    Required size is based on the model's internal RNN cell.

    Returns:
      A matrix of batch_size x cell size zeros.
    """
    return np.zeros((self.batch_size, self.cell.state_size))

  def restore_initialize_prime(self, session):
    """Saves the session, restores variables from checkpoint, primes model.

    Model is primed with its default midi file.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)
    self.prime_model()

  def initialize_and_restore(self, session):
    """Saves the session, restores variables from checkpoint.

    Args:
      session: A tensorflow session.
    """
    self.session = session
    self.restore_vars_from_checkpoint(self.checkpoint_dir)

  def initialize_new(self, session=None):
    """Saves the session, initializes all variables to random values.

    Args:
      session: A tensorflow session.
    """
    with self.graph.as_default():
      if session is None:
        self.session = tf.Session(graph=self.graph)
      else:
        self.session = session
      self.session.run(tf.initialize_all_variables())

  def get_variable_name_dict(self):
    """Constructs a dict mapping the checkpoint variables to those in new graph.

    Returns:
      A dict mapping variable names in the checkpoint to variables in the graph.
    """


    var_dict = dict()
    '''
    for var in self.variables():
      inner_name = rl_tuner_ops.get_inner_scope(var.name)
      inner_name = rl_tuner_ops.trim_variable_postfixes(inner_name)
      if '/Adam' in var.name:
        # TODO(lukaszkaiser): investigate the problem here and remove this hack.
        pass
      elif self.note_rnn_type == 'basic_rnn':
        var_dict[inner_name] = var
      else:
        var_dict[self.checkpoint_scope + '/' + inner_name] = var
    '''
    
    for var in self.variables():
      inner_name = rl_tuner_ops.get_inner_scope(var.name)
      inner_name = rl_tuner_ops.trim_variable_postfixes(inner_name)
      if '/Adam' in var.name:
        # TODO(lukaszkaiser): investigate the problem here and remove this hack.
        pass
      elif self.note_rnn_type == 'basic_rnn':
        var_dict[inner_name] = var
      else:
        var_dict[self.checkpoint_scope + '/' + inner_name] = var

    return var_dict

  def build_graph(self):
    """Constructs the portion of the graph that belongs to this model."""

    tf.logging.info('Initializing melody RNN graph for scope %s', self.scope)
    
    with self.graph.as_default():
      with tf.device(lambda op: ''):
        with tf.variable_scope(self.scope):
          # Make an LSTM cell with the number and size of layers specified in
          # hparams.
          if self.note_rnn_type == 'basic_rnn':
            self.cell = events_rnn_graph.make_rnn_cell(
                self.hparams.rnn_layer_sizes)
          else:
            self.cell = rl_tuner_ops.make_rnn_cell(self.hparams.rnn_layer_sizes,keep_prob=self.keep_prob,is_training=self.is_training)
          # Shape of melody_sequence is batch size, melody length, number of
          # output note actions.
          '''
          self.melody_sequence = tf.placeholder(tf.float32,
                                                [None, None,self.hparams.one_hot_length],name='melody_sequence')
                                                '''
          #batch size X melody length,
          self.melody_sequence = tf.placeholder(tf.int32,
                                                [None,None],name='melody_sequence')                                
          self.embed_seq = tf.contrib.layers.embed_sequence(self.melody_sequence, self.hparams.one_hot_length, self.hparams.rnn_layer_sizes[0])
          '''
          if self.is_training:
            self.embed_seq = tf.nn.dropout(self.embed_seq, keep_prob=self.keep_prob)
          '''
         # self.inputs = tf.nn.embedding_lookup(embedding, self.melody_sequence)
          self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
          self.initial_state = tf.placeholder(tf.float32,
                                              [None, self.cell.state_size],
                                              name='initial_state')
        

          # Closure function is used so that this part of the graph can be
          # re-run in multiple places, such as __call__.
          def run_network_on_melody(m_seq,
                                    lens,
                                    initial_state,
                                    swap_memory=True,
                                    parallel_iterations=1):
            """Internal function that defines the RNN network structure.

            Args:
              m_seq: A batch of melody sequences of one-hot notes.
              lens: Lengths of the melody_sequences.
              initial_state: Vector representing the initial state of the RNN.
              swap_memory: Uses more memory and is faster.
              parallel_iterations: Argument to tf.nn.dynamic_rnn.
            Returns:
              Output of network (either softmax or logits) and RNN state.
            """
            outputs, final_state = tf.nn.dynamic_rnn(
                self.cell,
                m_seq,
                sequence_length=lens,
                initial_state=initial_state,
                swap_memory=swap_memory,
                parallel_iterations=parallel_iterations)
            logits = tf.contrib.layers.fully_connected(outputs, self.hparams.one_hot_length, activation_fn=None)
            
            return logits, final_state
         
          (self.logits, self.state_tensor) = run_network_on_melody(
              self.embed_seq, self.lengths, self.initial_state)
          self.softmax = tf.nn.softmax(self.logits)
         # pdb.set_trace()
          self.run_network_on_melody = run_network_on_melody
        
        if self.training_file_list is not None:
          # Does not recreate the model architecture but rather uses it to feed
          # data from the training queue through the model.
          with tf.variable_scope(self.scope, reuse=True):
            zero_state = self.cell.zero_state(
                batch_size=self.hparams.batch_size, dtype=tf.float32)
            pdb.set_trace()
            (self.train_logits, self.train_state) = run_network_on_melody(
                self.train_sequence, self.train_lengths, zero_state)
            self.train_softmax = tf.nn.softmax(self.train_logits)
  def train(self,X,y,lengths,weights,num_of_epoch):
    self.training_cost=[]
    self.valid_perplexity=[]
    self.validation_cost=[]
    print ("start traininig")
    with self.graph.as_default():
      with tf.device(lambda op: ''):
        with tf.variable_scope(self.scope):
         sequence_length=self.hparams.sequence_len
         batch_size=self.hparams.batch_size
         self.ground_truth_y=tf.placeholder(tf.int32, [self.hparams.batch_size, sequence_length])
         self.weights=tf.placeholder(tf.float32, [self.hparams.batch_size, sequence_length])
         '''
         cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ground_truth_y))

         batch_loss = tf.reduce_mean(cost, name="batch_loss")
         #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
         global_step = tf.Variable(0, name='global_step', trainable=False)
         optimizer = tf.contrib.layers.optimize_loss(batch_loss, global_step, 1.0, "Adam",
                                    clip_gradients=5.0)
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.batch_loss=tf.contrib.seq2seq.sequence_loss(
        self.logits,
        self.ground_truth_y,
        self.weights,
        average_across_timesteps=True,
        average_across_batch=True)
        init = tf.global_variables_initializer()
        self.batch_loss = tf.reduce_sum(self.batch_loss)
        tvars = tf.trainable_variables()
        
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.batch_loss, tvars),
                                          5.0)
        self._lr = tf.Variable(1.0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        #optimizer =tf.train.AdamOptimizer(self._lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step(self.graph))
         
        self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr) 
        
        with tf.Session() as sess:
          # create initialized variables
          self.saver = tf.train.Saver(max_to_keep=10000000)
          sess.run(init)
          
          learning_rate=1.0
          for epoch in range(num_of_epoch):
            print ("epoch "+str(epoch)+'/'+str(num_of_epoch))
            avg_cost=0
            idx=range(len(X))
            random.shuffle(idx)
            K=range(len(X)/self.hparams.batch_size-1)
            num_of_batch_per_epoch=len(K)
            #pdb.set_trace()
            save_per_n_batch=1000
            for k in range((num_of_batch_per_epoch/save_per_n_batch)+1):
              start_batch=k*save_per_n_batch
              end_batch=min((k+1)*save_per_n_batch,num_of_batch_per_epoch)
              L=K[start_batch:end_batch]
             
              if(len(L)<=0): continue             
              sess.run(self._lr_update, feed_dict={self._new_lr: learning_rate})
              
              avg_cost=0
              for i in L:
                #print (i+1,"/",len(L))
                #start_time = time.time()
                initial_state=np.zeros((self.hparams.batch_size, self.cell.state_size))
                _,tmp_logits,initial_state, c = sess.run([self.train_op,
                  self.logits,self.state_tensor, self.batch_loss], 
                  feed_dict = {self.melody_sequence:  np.reshape(X[idx[i*batch_size:(i+1)*batch_size]],(batch_size,sequence_length)),
                  self.initial_state:initial_state,
                 self.ground_truth_y: np.reshape(y[idx[i*batch_size:(i+1)*batch_size]],(batch_size,sequence_length)), # ,
                 self.weights: np.reshape(weights[idx[i*batch_size:(i+1)*batch_size]],(batch_size,sequence_length)),
                 self.lengths:lengths[idx[i*batch_size:(i+1)*batch_size]]})
                avg_cost= avg_cost+c
                #print ("batch_time=",time.time()-start_time) # 0.25s
               
                
                
              avg_cost=avg_cost/len(L)
              print  ("Epoch:", (epoch+1),"nth_batch:",k, " cost =", "{:.5f}".format(avg_cost))
              if(epoch%1==0):
                save_loc = os.path.join(self.checkpoint_dir, 'Epoch'+str(epoch)+'_'+str(k)+'_'+str(avg_cost))
                self.saver.save(sess, save_loc)
              
                validation_avg_cost,perplexity=self.validation_evaluation(sess)
                self.validation_cost.append([epoch,k,validation_avg_cost])
                self.valid_perplexity.append([epoch,k,perplexity])
                self.training_cost.append([epoch,k,avg_cost])
                self.save_train_record()
                
            
  
  def save_train_record(self):
    with open(os.path.join(self.checkpoint_dir,'train_cost.txt'),'w') as f:
        for i in self.training_cost:
          f.write(str(i[0])+"_"+str(i[1])+":"+str(i[2])+'\n')
    with open(os.path.join(self.checkpoint_dir,'valid_perplexity.txt'),'w') as f:
        for i in self.valid_perplexity:
          f.write(str(i[0])+"_"+str(i[1])+":"+str(i[2])+'\n')
    with open(os.path.join(self.checkpoint_dir,'valid_cost.txt'),'w') as f:
        for i in self.validation_cost:
          f.write(str(i[0])+"_"+str(i[1])+":"+str(i[2])+'\n')



  def load_dictionary(self,dict_file_path):
    self.char_to_int={}
    self.int_to_char={}
    with open(dict_file_path,'r') as f:
      for line in f:
        tmp=line.strip('\r\n').split(':')
        tmp[0]=tmp[0].strip()
        self.char_to_int[tmp[0]]=int(tmp[1])
        self.int_to_char[int(tmp[1])]=tmp[0]

  def load_validationdata(self,valid_file_name):
    self.val_dataX=[]
    self.val_dataY=[]
    self.val_lengths=[]
    self.val_weights=[]

    max_length=self.hparams.sequence_len
    with open(valid_file_name,'r') as f:
        for line in f:
        
          line=line.lower()
          line=line.strip('\t\n')
          words=line.split()
          words.insert(0,'<start>')
          N=len(words)
          words.append('<end>')
          x=[]
          y=[]
          for i in range(N+1):
            x.append(myRNN.char_to_int.get(words[i],1))
            

          for k in range(1,len(x)):
            start_idx = max(0, k-max_length)
            tmpx=x[start_idx: k]
            tmpy=x[start_idx+1: k+1]
            tmpx.extend([myRNN.char_to_int['<end>']]*(max_length-len(tmpx)))
            tmpy.extend([myRNN.char_to_int['<end>']]*(max_length-len(tmpy)))   
            tmp_weights=[0]*max_length
            #print("k - start_idx",k - start_idx)
            tmp_weights[k - start_idx-1]=1

            self.val_lengths.append(k - start_idx)
            self.val_dataX.append(tmpx)
            self.val_dataY.append(tmpy)
            self.val_weights.append(tmp_weights)
  

  # compute perplexity
  def validation_evaluation(self,sess):
    batch_size=self.hparams.batch_size
    N=len(self.val_dataX)
    L=(N/self.hparams.batch_size-1)
    print ('validation_evaluation number of batch=',L)
    perplexity=0.0
    avg_cost=0
    for i in range(L):
      #start_time=time.time()
      x=self.val_dataX[i*batch_size:(i+1)*batch_size]
      y=self.val_dataY[i*batch_size:(i+1)*batch_size]
      length=self.val_lengths[i*batch_size:(i+1)*batch_size]
      w=self.val_weights[i*batch_size:(i+1)*batch_size]
      initial_state=np.zeros((self.hparams.batch_size, self.cell.state_size))
      sess.run(self._lr_update, feed_dict={self._new_lr: 0.0})
      batch_loss,tmp_logits,initial_state,  = sess.run([self.batch_loss,
          self.logits,self.state_tensor ], 
          feed_dict = {self.melody_sequence:  x,
          self.initial_state:initial_state,
         self.ground_truth_y: y, 
         self.weights: w,
         self.lengths:length})
      avg_cost= avg_cost+batch_loss
      #print ("batch_time2=",time.time()-start_time) # 0.08s
      perplexity=perplexity-batch_loss
    perplexity=perplexity/L
    avg_cost=avg_cost/L
    perplexity=np.exp(-perplexity)
    print ("valid_avg_cost=",avg_cost)
    print ("perplexity=",perplexity)
    return avg_cost,perplexity

      

      

  def restore_vars_from_checkpoint(self, checkpoint_dir):
    """Loads model weights from a saved checkpoint.

    Args:
      checkpoint_dir: Directory which contains a saved checkpoint of the
        model.
    """
    tf.logging.info('Restoring variables from checkpoint')
    
    var_dict = self.get_variable_name_dict()
    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)

    tf.logging.info('Checkpoint dir: %s', checkpoint_dir)
    
    with self.graph.as_default():
      saver = tf.train.Saver(var_list=var_dict)
      checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
      if checkpoint_file is None:
        tf.logging.warn("Can't find checkpoint file, using %s",
                        self.checkpoint_file)
        checkpoint_file = self.checkpoint_file
      tf.logging.info('Checkpoint file: %s', checkpoint_file)
      print ('Checkpoint file:', checkpoint_file)
      
      saver.restore(self.session, checkpoint_file)

  def load_primer(self):
    """Loads default MIDI primer file.

    Also assigns the steps per bar of this file to be the model's defaults.
    """

    if not os.path.exists(self.midi_primer):
      tf.logging.warn('ERROR! No such primer file exists! %s', self.midi_primer)
      return

    self.primer_sequence = midi_io.midi_file_to_sequence_proto(self.midi_primer)
    quantized_seq = sequences_lib.quantize_note_sequence(
        self.primer_sequence, steps_per_quarter=4)
    extracted_melodies, _ = melodies_lib.extract_melodies(quantized_seq,
                                                          min_bars=0,
                                                          min_unique_pitches=1)
    self.primer = extracted_melodies[0]
    self.steps_per_bar = self.primer.steps_per_bar

  def prime_model(self):
    """Primes the model with its default midi primer."""
    with self.graph.as_default():
      tf.logging.debug('Priming the model with MIDI file %s', self.midi_primer)

      # Convert primer Melody to model inputs.
      encoder = magenta.music.OneHotEventSequenceEncoderDecoder(
          magenta.music.MelodyOneHotEncoding(
              min_note=rl_tuner_ops.MIN_NOTE,
              max_note=rl_tuner_ops.MAX_NOTE))

      seq = encoder.encode(self.primer)
      features = seq.feature_lists.feature_list['inputs'].feature
      primer_input = [list(i.float_list.value) for i in features]

      # Run model over primer sequence.
      primer_input_batch = np.tile([primer_input], (self.batch_size, 1, 1))
      self.state_value, softmax = self.session.run(
          [self.state_tensor, self.softmax],
          feed_dict={self.initial_state: self.state_value,
                     self.melody_sequence: primer_input_batch,
                     self.lengths: np.full(self.batch_size,
                                           len(self.primer),
                                           dtype=int)})
      priming_output = softmax[-1, :]
      self.priming_note = self.get_note_from_softmax(priming_output)

  def get_note_from_softmax(self, probs,top_n=5):
    """Extracts a one-hot encoding of the most probable note.

    Args:
      softmax: Softmax probabilities over possible next notes.
    
    """


    '''
    probs = np.array(probs, dtype=np.float64)
    # set probabilities after top_n to 0
    probs[np.argsort(probs)[:-top_n]] = 0
    # renormalise probabilities
    probs /= np.sum(probs)
    sampled_index = np.random.choice(len(probs), p=probs)
    return sampled_index

    '''
    return np.argmax(probs)
    #return np.random.choice(len(probs), 1, p=probs)[0]
    



  def __call__(self):
    """Allows the network to be called, as in the following code snippet!

        q_network = MelodyRNN(...)
        q_network()

    The q_network() operation can then be placed into a larger graph as a tf op.

    Note that to get actual values from call, must do session.run and feed in
    melody_sequence, lengths, and initial_state in the feed dict.

    Returns:
      Either softmax probabilities over notes, or raw logit scores.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        logits, self.state_tensor = self.run_network_on_melody(
            self.embed_seq, self.lengths, self.initial_state)
        return logits

  def run_training_batch(self):
    """Runs one batch of training data through the model.

    Uses a queue runner to pull one batch of data from the training files
    and run it through the model.

    Returns:
      A batch of softmax probabilities and model state vectors.
    """
    if self.training_file_list is None:
      tf.logging.warn('No training file path was provided, cannot run training'
                      'batch')
      return
    print ('run_training_batch')
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=self.session, coord=coord)
    pdb.set_trace()
    softmax, state, lengths = self.session.run([self.train_softmax,
                                                self.train_state,
                                                self.train_lengths])
    
    coord.request_stop()

    return softmax, state, lengths

  def get_next_note_from_note(self, note,length):
    """Given a note, uses the model to predict the most probable next note.

    Args:
      note: A one-hot encoding of the note.
    Returns:
      Next note in the same format.
    """
    with self.graph.as_default():
      with tf.variable_scope(self.scope, reuse=True):
        

        input_batch = np.reshape(note,
                                 (1, -1))
        #self.state_value =np.zeros((1, self.cell.state_size))
        softmax, self.state_value = self.session.run(
            [self.softmax, self.state_tensor],
            {self.melody_sequence: input_batch,
             self.initial_state: self.state_value,
             self.lengths:[length]})
        
        return self.get_note_from_softmax(softmax[0][length-1])

  def variables(self):
    """Gets names of all the variables in the graph belonging to this model.

    Returns:
      List of variable names.
    """
    with self.graph.as_default():
      return [v for v in tf.global_variables() if v.name.startswith(self.scope)]
  

if __name__ =='__main__':
  graph = tf.Graph()
  hparams= rl_tuner_ops.default_hparams()
  optimizer = tf.train.AdamOptimizer()
  myRNN =NoteRNNLoader(
          graph, 'q_network',
          checkpoint_dir='./cpk',
          checkpoint_file=None,
          midi_primer=None,
          training_file_list=None,
          hparams=hparams,
          note_rnn_type='default',
          is_training=True)
  dict_path='./ptb_dict.txt'
  myRNN.load_dictionary(dict_path)
  myRNN.load_validationdata("./ptb.valid.txt")
  filename = "./ptb.train.txt"
  dataX=[]
  dataY=[]
  lengths=[]
  weights=[]
  max_length=hparams.sequence_len
  with open(filename,'r') as f:
      for line in f:
      
        line=line.lower()
        line=line.strip('\t\n')
        words=line.split()
        words.insert(0,'<start>')
        N=len(words)
        words.append('<end>')
        x=[]
        y=[]
        for i in range(N):
          x.append(myRNN.char_to_int.get(words[i],1))
          y.append(myRNN.char_to_int.get(words[i+1],1))
        if(len(x)>max_length):
         
          for k in range(len(x)-max_length + 1):
            tmpx=x[k:k+max_length]
            tmpy=y[k:k+max_length]
            
  
            weight=[1]*max_length
            weights.append(weight)
            lengths.append(max_length)
            dataX.append(tmpx)
            dataY.append(tmpy)
          
        else:
          lengths.append(len(x)) 
          x.extend([myRNN.char_to_int['<end>']]*(max_length-len(x)))
          y.extend([myRNN.char_to_int['<end>']]*(max_length-len(y))) 
          weight=[1]*lengths[-1]+[0]*(max_length-lengths[-1])
          dataX.append(x)
          dataY.append(y)
          weights.append(weight)
       

        
  
  dataX=np.array(dataX)
  dataY=np.array(dataY)
  lengths=np.array(lengths)
  weights=np.array(weights)
  '''
  dataX=dataX[0:1000]
  dataY=dataY[0:1000]
  lengths=lengths[0:1000]
  weights=weights[0:1000]
  '''
  print ('max_length',max_length)
  

  '''
  # create mapping of unique chars to integers
  chars = sorted(list(set(raw_text)))
  char_to_int = dict((c, i) for i, c in enumerate(chars))
  int_to_char = dict((i, c) for i, c in enumerate(chars))
  '''
  # summarize the loaded data
  
  n_patterns = len(dataX)
  print ("Total Patterns: ", n_patterns)
  X = np.reshape(dataX, (n_patterns, max_length, 1))
  y = np.reshape(dataY, (n_patterns, max_length, 1))
  #myRNN.train(dataX,dataY,lengths,weights,1000000)
  
  #exit(0)
  pdb.set_trace()
  graph2= tf.Graph()
  myRNN2 =NoteRNNLoader(
          graph2, 'q_network2',
          checkpoint_dir='./ckp1',
          checkpoint_file=None,
          midi_primer=None,
          training_file_list=None,
          hparams=hparams,
          note_rnn_type='default',
          is_training=False,
          checkpoint_scope='q_network')

  tmp_session=tf.Session(graph=graph2)


  
  myRNN2.initialize_and_restore(tmp_session)
  
  
  
  count=1
  generated_text=myRNN.int_to_char[x[0]]



  dataset=myRNN.val_dataX
  L=range(len(dataset))
  random.shuffle(L)
  for i in L[0:50]:
    '''
    if(dataset[i][0]==2):
      x=[dataset[i][1]]
    else:
      x=[dataset[i][0]]
    '''
    x=np.array([np.random.randint(0, rl_tuner_ops.NUM_CLASSES - 1)])
    generated_text=myRNN.int_to_char[x[0]]
    count=1
    while(True):
      
      idx=myRNN2.get_next_note_from_note([x],count)
      x[0]=idx
      
      
      
      generated_text+=' '+myRNN.int_to_char[idx]
     
      if(myRNN.int_to_char[idx]=='<end>'): 
        myRNN2.state_value =np.zeros((1, myRNN2.cell.state_size))
        break
      
    print (generated_text)
  pdb.set_trace()






