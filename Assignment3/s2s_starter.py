import torch
import numpy as np
from torch import nn
import copy
import torch.nn.functional as F
# class for our vanilla seq2seq

# class for our vanilla seq2seq
class S2S(nn.Module):
    def __init__(self, d_char, d_hid, len_voc):

        super(S2S, self).__init__()
        self.d_char = d_char
        self.d_hid = d_hid
        self.len_voc = len_voc

        # embeddings
        self.char_embs = nn.Embedding(len_voc, d_char)

        # encoder and decoder RNNs
        self.encoder = nn.RNN(d_char, d_hid, num_layers=1, batch_first=True)
        self.decoder = nn.RNN(d_char, d_hid, num_layers=1, batch_first=True)
        
        # output layer (softmax will be applied after this)
        self.out = nn.Linear(d_hid, len_voc)
        self.concat=nn.Linear(2*d_hid, d_hid)

    
    # perform forward propagation of S2S model
    def forward(self, inputs, outputs):
        
        bsz, max_len = inputs.size()

        # get embeddings of inputs
        embs = self.char_embs(inputs)

        # encode input sentence and extract final hidden state of encoder RNN
        encoder_outputs, final_enc_hiddens = self.encoder(embs)
                
        # initialize decoder hidden to final encoder hiddens
        hn = final_enc_hiddens
        
        # store all decoder states in here
        decoder_states = torch.zeros(max_len, bsz, self.d_hid)

        # now decode one character at a time
        for idx in range(max_len):

            # store the previous character if there was one
            prev_chars = None 
            if idx > 0:
                prev_chars = outputs[:, idx - 1] # during training, we use the ground-truth previous char

            # feed previous hidden state and previous char to decoder to get current hidden state
            if idx == 0:
                decoder_input = torch.zeros(bsz, 1, self.d_char)

            # get previous ground truth char embs
            else:
                decoder_input = self.char_embs(prev_chars)
                decoder_input = decoder_input.view(bsz, 1, self.d_char)
                
                
                
                
            # feed to decoder rnn and store hidden state in decoder states
            _, hn = self.decoder(decoder_input, hn)
            #decoder_outputs.transpose(0,1) is equal to hn
            ###################################################################
            #                         ATTENTION                               #
            ###################################################################
            
            ##print("dec-enc out put ",decoder_outputs.shape,"---------", encoder_outputs.shape) 
            ##print("dec-enc hidden ",hn.shape,"---------",final_enc_hiddens.shape)
            #print(hn.transpose(0,1)[0][0] ,"************",encoder_outputs[0][0])
            #attn_mult=hn.transpose(0,1) * encoder_outputs
            #print("===============================================\n",attn_mult[0][0])
            
            ##attention_weights=torch.sum(hn.transpose(0,1) * encoder_outputs, dim=2)
            ##print('attention_scores=============',attention_weights.shape)
            ##softmax=F.softmax(attention_weights, dim=1).unsqueeze(2)
            ##print('attention_scores=============',softmax.shape)
            
            #print(attention_weights[0][0] ,"************",encoder_outputs[0][0])
            ##m=softmax*encoder_outputs
            #print("===============================================\n",m)
            #3context = torch.sum(m,dim=1).unsqueeze(1)
            
            ##print("context ", context.shape)
            
            
            # Concatenate weighted context vector and decoder output using Luong eq. 5
            #decoder_outputs = decoder_outputs.squeeze(0)
            #context = context.squeeze(1)
            ##concat = self.concat(  torch.cat((hn, context.transpose(0,1)), 2)  )
            #concat_output = self.concat(concat)
   
            #concat=torch.cat((torch.unsqueeze(context,1), decoder_outputs), 2)
            ##print("attention result: " ,concat.shape)
        
            ##hn = concat
            #####################################################################
            #                              ATTENTION                            #
            #####################################################################
            decoder_states[idx] =hn #concat#

        # now do prediction over decoder states (reshape to 2d first)
        decoder_states =  decoder_states.transpose(0, 1).contiguous().view(-1, self.d_hid)
        #print(decoder_states.shape)
        decoder_preds = self.out(decoder_states)
        #print(decoder_preds.shape)
        decoder_preds = torch.nn.functional.log_softmax(decoder_preds, dim=1)
        #print(decoder_preds.shape)
        return decoder_preds
  
    # given a previous character and a previous hidden state
    # produce a probability distribution over the entire vocabulary
    def single_decoder_step(self, prev_char, prev_hid,encoder_outputs):
        if prev_char is not None:
            decoder_input = self.char_embs(prev_char).expand(1, 1, self.d_char)
        else:
            decoder_input = torch.zeros(1, 1, self.d_char)

        # feed prev hidden state and prev char to decoder to get current hidden state
        _, hn = self.decoder(decoder_input, prev_hid)

        #####################################################################
        #                              ATTENTION                            #
        #####################################################################
        ##attention_weights=torch.sum(hn.transpose(0,1) * encoder_outputs, dim=2)
            ##print('attention_scores=============',attention_weights.shape)
        #3softmax=F.softmax(attention_weights, dim=1).unsqueeze(2)
            ##print('attention_scores=============',softmax.shape)
            
            #print(attention_weights[0][0] ,"************",encoder_outputs[0][0])
        ##m=softmax*encoder_outputs
            #print("===============================================\n",m)
        ##context = torch.sum(m,dim=1).unsqueeze(1)
            
            ##print("context ", context.shape)
            
            
            # Concatenate weighted context vector and decoder output using Luong eq. 5
            #decoder_outputs = decoder_outputs.squeeze(0)
            #context = context.squeeze(1)
        ##concat = self.concat(  torch.cat((hn, context.transpose(0,1)), 2)  )
            #concat_output = self.concat(concat)
   
            #concat=torch.cat((torch.unsqueeze(context,1), decoder_outputs), 2)
            ##print("attention result: " ,concat.shape)
        ##hn = concat
            
            
        
        #####################################################################
        #                              ATTENTION                            #
        #####################################################################
        
        # feed into output layer and apply softmax to get probability distribution over vocab
        pred_dist = self.out(hn.transpose(0, 1).contiguous().view(-1, self.d_hid))
        pred_dist = torch.nn.functional.log_softmax(pred_dist, dim=1)
        return pred_dist.view(-1), hn


    # greedy search for one input sequence (bsz = 1)
    def greedy_search(self, seq):

        bsz, max_len = seq.size()

        output_seq = [] # this will contain our output sequence
        output_prob = 0. # and this will be the probability of that sequence

        # get embeddings of inputs
        embs = self.char_embs(seq)

        # encode input sentence and extract final hidden state of encoder RNN
        encoder_outputs, final_enc_hidden = self.encoder(embs)

        # initialize decoder hidden to final encoder hidden
        hn = final_enc_hidden
        prev_char = None
        
        # now decode one character at a time
        for idx in range(max_len):

            pred_dist, hn = self.single_decoder_step(prev_char, hn,encoder_outputs)

            _, top_indices = torch.sort(-pred_dist) # sort in descending order (log domain)
            # in greedy search, we will just use the argmax prediction at each time step
            argmax_pred = top_indices[0]
            argmax_prob = pred_dist[argmax_pred]
            output_seq.append(argmax_pred.numpy())
            output_prob += argmax_prob

            # now use the argmax prediction as the input to the next time step
            prev_char = argmax_pred

        return output_prob, output_seq


    # beam search for one input sequence (bsz = 1)
    # YOUR JOB: FILL THIS OUT!!! SOME SPECIFICATION:
    # input: seq (a single input sequence)
    # output: beams, a list of length beam_size containing the most probable beams
    #         each element of beams is a tuple whose first two elements are
    #         the probability of the sequence and the sequence itself, respectively.
    def beam_search(self, seq, beam_size=5):
        #print("start beam search ...")
        bsz, max_len = seq.size()

        # get embeddings of inputs
        embs = self.char_embs(seq)

        # encode input sentence and extract final hidden state of encoder RNN
        encoder_outputs, final_enc_hidden = self.encoder(embs)
        
        # this list will contain our k beams
        # you might find it helpful to store three things:
        #       (prob of beam, all chars in beam so far, prev hidden state)
        beams = [(0.0, [], final_enc_hidden)]  #instead of previous char in greedy algorithm

        # decode one character at a time
        for idx in range(max_len):
            #print("create next character-->", idx,"/",max_len)
            # add all candidate beams to the below list
            # later you will take the k most probable ones where k = beam_size
            beam_candidates = []
      
            stop=0
            for b in beams:
                #print("\n-----------------beam------------\n: ",b,len(beams))
                curr_prob, seq, prev_h = b

                if len(seq) == 0:
                    prev_char = None
                else:
                    prev_char = seq[-1]

                #print("prev char: ",prev_char)
                pred_dist, prev_h = self.single_decoder_step(prev_char, prev_h,encoder_outputs)
                _, top_indices = torch.sort(-pred_dist) # sort in descending order (log domain)
                #print("pred distance: ",pred_dist.tolist())
                #print("top indecies: ",top_indices.tolist())
                
                 # in greedy search, we will just use the argmax prediction at each time step
                argmax_pred = top_indices[:beam_size]
                #print("prediction:   " ,argmax_pred)
                argmax_prob=pred_dist[argmax_pred]
                #argmax_prob=[]
                #for i in argmax_pred:
                    #argmax_prob.append(pred_dist[i])
                #print("probability:   " ,argmax_prob)
            #output_seq.append(argmax_pred.numpy())
            #output_prob += argmax_prob

            # now use the argmax prediction as the input to the next time step
            #prev_char = argmax_pred
          
                
                for i in range(beam_size):
                    #print("in loop to add other chars - argmax prediction: ",i,argmax_pred[i])
                    seq_=[]
                    seq_=copy.deepcopy(seq)
                    #print(seq_,seq)
                    seq_.append(argmax_pred[i])
                    tuple_ = ((argmax_prob[i]+curr_prob).tolist(), seq_, prev_h)
                    #print("beam candidate before append: ",beam_candidates)
                    beam_candidates.append(tuple_)
                    #print("beam candidate after append: ",beam_candidates)
                    #print(tuple_)
            #print("--------------------------")
            
            #print("not sorted : ",beam_candidates)
            #beam_candidates=sorted(beam_candidates, key=lambda x: x[0])
            #print(beam_candidates[0][0].item())           
            
            beam_candidates.sort(key=lambda x: -x[0])
            #print("Sorted ####----------##### :",beam_candidates)
            beams = beam_candidates[:beam_size]
            #print("current beams: ",beams)

            #return output_prob, output_seq

        # fill out the rest of the beam search!
        # the greedy_search code might be helpful to look at and understand!

        return beams


    def beam_check(self, seq):
        beams = self.beam_search(seq.expand(1, seq.size()[1]), beam_size=1)
        greedy_prob, greedy_out = self.greedy_search(seq.expand(1, seq.size()[1]))
        beam_prob = beams[0][0]
        beam_out = [np.array(c) for c in beams[0][1]]

        if beams[0][0] == greedy_prob and greedy_out == beam_out:
            return True
        else: 
            return False


