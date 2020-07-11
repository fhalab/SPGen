import time

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim

import numpy as np

from transformer import Models
from transformer import Beam
from transformer import Translator
from transformer.Optim import ScheduledOptim

import sys

pad = 0


class ScheduledOptimizer(object):
    '''A simple wrapper class for learning rate scheduling. '''

    def __init__(self, optimizer, n_warmup_steps, d_model=None, lr_max=5e-3,
                 decay_power=-0.5):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.decay_power = decay_power
        if self.d_model is not None:
            self.front_factor = np.power(self.d_model, -0.5)
            if n_warmup_steps > 0:
                self.lr_max = self.front_factor * np.power(n_warmup_steps,
                                                           decay_power)
            else:
                self.lr_max = self.front_factor

        else:
            self.lr_max = lr_max
            if n_warmup_steps > 0:
                self.front_factor = lr_max / np.power(n_warmup_steps, decay_power)
            else:
                self.front_factor = lr_max
        if n_warmup_steps > 0:
            self.factor = np.power(n_warmup_steps, -1 + decay_power)
        else:
            self.factor = 1e10

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def test_learning_rate(x):
        return [self.get_learning_rate()]

    def get_learning_rate(self, steps):
        return self.front_factor * np.minimum(
            np.power(steps + 1, self.decay_power),
            self.factor * (steps + 1))

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_learning_rate(self.n_current_steps)


class SignalTranslator(object):

    def __init__(self, model_opt, optim_opt, trans_opt):
        self.model_opt = model_opt
        self.model = Models.Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            proj_share_weight=model_opt.proj_share_weight,
            embs_share_weight=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner_hid=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)
        self.n_params = sum(np.prod(p.size()) for p in self.model.parameters())
        print('Initiated Transformer with %d parameters.' %self.n_params)

        self.optim_opt = optim_opt
        optim = optim_opt.optim(self.model.get_trainable_parameters())
        self.optimizer = ScheduledOptimizer(
            optim,
            optim_opt.n_warmup_steps,
            lr_max=optim_opt.lr_max,
            d_model=optim_opt.d_model,
            decay_power=optim_opt.decay_power)

        self.crit = self._get_criterion()

        self.trans_opt = trans_opt
        self.cuda = model_opt.cuda

        # Need this for translating
        prob_projection = nn.LogSoftmax()
        self.tt = torch.cuda if self.cuda else torch

        if self.cuda:
            self.model = self.model.cuda()
            self.crit = self.crit.cuda()
            prob_projection.cuda()

        self.model.prob_projection = prob_projection

        self.history = {
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[]
        }
        self.epochs = 0

    def _get_criterion(self):
        weight = torch.ones(self.model_opt.tgt_vocab_size)
        return nn.CrossEntropyLoss(weight, size_average=False)

    def _get_performance(self, pred, gold):
        loss = self.crit(pred, gold.contiguous().view(-1))
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(pad).data).sum()
        return loss, n_correct

    @staticmethod
    def generator_from_h5(h5file, batch_size, use_cuda=False,
                          shuffle=True, return_y=True):
        n, L_y = h5file['y'].shape
        _, L_X = h5file['X'].shape
        inds = np.arange(n)
        batches = int(np.ceil(n / batch_size))
        if shuffle:
            np.random.shuffle(inds)
        for idx in range(batches):
            if idx == batches - 1:
                batch_idxs = inds[idx * batch_size:]
            else:
                batch_idxs = inds[idx * batch_size: (idx + 1) * batch_size]
            batch_idxs = sorted(batch_idxs)
            X = h5file['X'][batch_idxs, :]
            X = Variable(torch.from_numpy(X).long())
            pos_X = np.array([[i + 1 for i in range(L_X)]
                              for j in range(len(batch_idxs))])
            pos_X = Variable(torch.LongTensor(pos_X))
            if return_y:
                y = h5file['y'][batch_idxs, :]
                # Make y into tensor
                y = Variable(torch.FloatTensor(y).long())
                pos_y = np.array([[i + 1 for i in range(L_y)]
                                  for j in range(len(batch_idxs))])
                pos_y = Variable(torch.LongTensor(pos_y))
                if use_cuda:
                    X = X.cuda()
                    pos_X = pos_X.cuda()
                    y = y.cuda()
                    pos_y = pos_y.cuda()
                yield (X, pos_X), (y, pos_y)
            else:
                if use_cuda:
                    X = X.cuda()
                    pos_X = pos_X.cuda()
                yield X, pos
                
    @staticmethod
    def generator_from_h5_noy(h5file, batch_size, use_cuda=False,
                          shuffle=True):
        """ Generates batches from dataset with no y """
        n, L_X = h5file['X'].shape
        inds = np.arange(n)
        batches = int(np.ceil(n / batch_size))
        if shuffle:
            np.random.shuffle(inds)
        for idx in range(batches):
            if idx == batches - 1:
                batch_idxs = inds[idx * batch_size:]
            else:
                batch_idxs = inds[idx * batch_size: (idx + 1) * batch_size]
            batch_idxs = sorted(batch_idxs)
            X = h5file['X'][batch_idxs, :]
            X = Variable(torch.from_numpy(X).long())
            pos_X = np.array([[i + 1 for i in range(L_X)]
                              for j in range(len(batch_idxs))])
            pos_X = Variable(torch.LongTensor(pos_X))
            
            if use_cuda:
                X = X.cuda()
                pos_X = pos_X.cuda()
            yield X, pos_X

    def _epoch(self, data, train):
        ''' Epoch operation'''

        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0
        n_total_words = 0
        n_total_correct = 0
        for i, batch in enumerate(data):
            # prepare data
            
            src, tgt = batch
            gold = tgt[0][:, 1:]

            # forward
            if train:
                self.optimizer.zero_grad()
            pred = self.model(src, tgt)
            loss, n_correct = self._get_performance(pred, gold)

            if train:
                # backward
                loss.backward()
                # update parameters
                self.optimizer.step()
                self.optimizer.update_learning_rate()

            # note keeping
            n_words = gold.data.ne(pad).sum()
            n_total_words += n_words
            n_total_correct += n_correct
            total_loss += loss.data[0]
            L = total_loss / n_total_words
            acc = n_total_correct / n_total_words
            if train:
                update = '\rTraining batch %d\tloss: %.4f\tacc: %.4f' \
                    %(i + 1, L, acc)
            else:
                update = '\rValidation batch %d\tloss: %.4f\tacc: %.4f' \
                    %(i + 1, L, acc)
            print(update, end='')
        self.epochs += 1
        return L, acc

    def train(self, train_file, val_file, batch_size=32,
              val_batch_size=64, epochs=20, **kwargs):
        epoch_start = self.epochs
        for epoch_i in range(epochs):

            print('Epoch %d of %d ' %(epoch_i + 1 + epoch_start,
                                      epochs + epoch_start))

            train_data = SignalTranslator.generator_from_h5(train_file,
                                                            batch_size,
                                                            use_cuda=self.cuda)
            start = time.time()
            
            train_loss, train_accu = self._epoch(train_data, True)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accu)
            print('\ttime: {elapse:3.3f} s'.format(elapse=(time.time() - start)))

            if val_file is not None:
                start = time.time()
                val_data = SignalTranslator.generator_from_h5(val_file,
                                                              val_batch_size,
                                                              shuffle=False,
                                                              use_cuda=self.cuda)
                valid_loss, valid_accu = self._epoch(val_data, False)
                print('\ttime: {elapse:3.3f} s'.format(elapse=(time.time()-start)))
                self.history['val_loss'].append(valid_loss)
                self.history['val_acc'].append(valid_accu)

            if 'save_model' in kwargs:
                model_name = kwargs['save_model']
                if kwargs['save_mode'] == 'all':
                    model_name += '_{epoch:d}.chkpt'.format(epoch=epoch_i)
                    self.save_model(model_name)
                elif kwargs['save_mode'] == 'best':
                    if valid_loss <= min(self.history['val_loss']):
                        model_name += '.chkpt'
                        self.save_model(model_name)
                        print('The chkpt file has been updated at epoch %d.'
                              %(epoch_i + 1))
                        
            #### stop early if validation loss >> training loss          
            #training_loss = self.history['train_loss'].values()   
            #validation_loss = self.history['val_loss'].values()       
            #if all(x<y for x, y in zip(validation_loss, validation_loss[-5:])) and all(x>y for x, y in zip(training_loss, training_loss[-5:])): 
                #sys.exit()
        return self.history

    def save_model(self, fname):
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': (self.model_opt, self.trans_opt, self.optim_opt)}
        torch.save(checkpoint, fname)

    @classmethod
    def load_model(cls, fname):
        checkpoint = torch.load(fname)
        model_opt, trans_opt, optim_opt = checkpoint['settings']
        print(model_opt, trans_opt, optim_opt)
        clf = cls(model_opt, optim_opt, trans_opt)
        clf.model.load_state_dict(checkpoint['model'])
        return clf

    def translate_batch(self, src_batch, beam):
        self.model.eval()
        
        # Beam size, or beam width, is a parameter in the beam search algorithm which determines how many of 
        # the best partial solutions to evaluate. In an LSTM model of melody generation, for example, beam size limits the number of 
        # candidates to take as input for the decoder. A beam size of 1 is a best-first search - only the most probable candidate is chosen 
        # as input for the decoder. A beam size of k will decode and evaluate the top k candidates. A large beam size means a more 
        # extensive search - not only the single best candidate is evaluated.
        
        # Batch size is in different location depending on data.
        src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        beam_size = beam
        # beam_size = self.trans_opt.beam_size
#         print("beam_size")
#         print(beam_size)

        # Encode
        enc_outputs, enc_slf_attns = self.model.encoder(src_seq, src_pos) # enc_outputs, beam search, decoder
        
        # Repeat data for beam
        src_seq = Variable(src_seq.data.repeat(beam_size, 1))
        enc_outputs = [Variable(enc_output.data.repeat(beam_size, 1, 1)) for enc_output in enc_outputs]

        # Prepare beams
        beam = [Beam.Beam(beam_size, self.cuda) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        n_remaining_sents = batch_size
        
        # A larger beam generally means a more accurate prediction at the expense of memory and time
        # Beam search is a heuristic search algorithm that uses breadth-first search to build its search tree and reduces the search space 
        # by eliminating candidates to reduce the memory and time requirements.
        
        # Decode
        for i in range(self.trans_opt.max_trans_length):

            len_dec_seq = i + 1

            # -- Preparing decode data seq -- #
            input_data = torch.stack([
                b.get_current_state() for b in beam if not b.done]) # size: mb x bm x sq
            input_data = input_data.view(-1, len_dec_seq)           # size: (mb*bm) x sq
            input_data = Variable(input_data, volatile=True)

            # -- Preparing decode pos seq -- #
            # size: 1 x seq
            input_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: (batch * beam) x seq
            input_pos = input_pos.repeat(n_remaining_sents * beam_size, 1)
            input_pos = Variable(input_pos.type(torch.LongTensor),
                                 volatile=True)

            if self.cuda:
                input_pos = input_pos.cuda()
                input_data = input_data.cuda()

            # -- Decoding -- #
            dec_outputs, dec_slf_attns, dec_enc_attns = self.model.decoder(
                input_data, input_pos, src_seq, enc_outputs)
            dec_output = dec_outputs[-1][:, -1, :]  # (batch * beam) * d_model
            dec_output = self.model.tgt_word_proj(dec_output)
            out = self.model.prob_projection(dec_output)

            # batch x beam x n_words
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active_enc_info(tensor_var, active_idx):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                tensor_data = tensor_var.data.view(n_remaining_sents, -1, self.model_opt.d_model)

                new_size = list(tensor_var.size())
                new_size[0] = new_size[0] * len(active_idx) // n_remaining_sents

                # select the active index in batch
                return Variable(
                    tensor_data.index_select(0, active_idx).view(*new_size),
                    volatile=True)

            def update_active_seq(seq, active_idx):
                ''' Remove the src sequence of finished instances in one batch. '''
                view = seq.data.view(n_remaining_sents, -1)
                new_size = list(seq.size())
                new_size[0] = new_size[0] * len(active_idx) // n_remaining_sents # trim on batch dim

                # select the active index in batch
                return Variable(view.index_select(0, active_idx).view(*new_size), volatile=True)

            src_seq = update_active_seq(src_seq, active_idx)
            enc_outputs = [
                update_active_enc_info(enc_output, active_idx)
                for enc_output in enc_outputs]
            n_remaining_sents = len(active)

        # Return useful information
        all_hyp, all_scores = [], []
        n_best = self.trans_opt.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [beam[b].get_hypothesis(k) for k in ks[:n_best]]
            all_hyp += [hyps]

        decoded = [self.trans_opt.ctable.decode(hyps[0]) for hyps in all_hyp] # all_hyp

        return decoded, all_hyp, all_scores, enc_outputs, dec_outputs, enc_slf_attns, dec_slf_attns, dec_enc_attns
