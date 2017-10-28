from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import onmt
from onmt.Utils import aeq
import sys
import numpy as np

class VaeEncoder(nn.Module):
    """ The standard RNN encoder. """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings, z_size):
        super(VaeEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False
        self.varcoeff = 0.0
        self.varstep = 0.1

        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

        self.h2z = nn.Linear(hidden_size, z_size * 2)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        #new_h = [torch.cat([hidden_t[i][0:h.size(0):2], hidden_t[i][1:h.size(0):2]], 2) for i in range(len(hidden_t))]
        hh = hidden_t[0] #in LSTM, hidden_t[0] is hidden states, while hidden_t[1] is cell states
        #hh.size(), num_layers * num_directions, batch, hidden_size
        hhh = torch.mean(hh,0)


        ps = self.h2z(hhh)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)

        #return hidden_t, outputs
        #return mu, logvar, z, hidden_t, outputs
        return outputs, z,  mu, logvar

    def manip(self, input, steps, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        #new_h = [torch.cat([hidden_t[i][0:h.size(0):2], hidden_t[i][1:h.size(0):2]], 2) for i in range(len(hidden_t))]
        hh = hidden_t[0] #in LSTM, hidden_t[0] is hidden states, while hidden_t[1] is cell states
        #hh.size(), num_layers * num_directions, batch, hidden_size
        hhh = torch.mean(hh,0)  # batch, hidden_size


        ps = self.h2z(hhh)   # batch, 2*z_size
        mu, logvar = torch.chunk(ps, 2, dim=1)
        #z = self.sample(mu, logvar)  #batch, z_size
        z = self.controlled_change(mu, steps) 

        for k in range(steps):
            outputs = torch.cat((outputs, outputs),1)

        return outputs, z

    def controlled_change(self, mu, steps):
        # want to change z linearly from one point to the other point
        # after this the batchsize  will populate more. 
        mu = mu.transpose(0,1)
        z = None
        for i in range(mu.size()[0]):
            for j in range(steps):
                if i==mu.size()[0]-1:
                    newz = mu[i]
                else:
                    newz = mu[i] + j/steps*mu[i+1]
                newz = newz.unsqueeze(0)
                if(z is None):
                    z = newz
                else:
                    z = torch.cat((z,newz),0)  
        z = torch.cat((z,newz),0) 
        return z.transpose(0,1)



    def sample(self, mu, logvar):
        #print(mu.size())
        #print(logvar.size())
        eps = Variable(torch.randn(logvar.size())).cuda() #.cuda()
        #eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        #print(eps.data.type())
        #print(std.data.type())
        z = mu + eps*std*self.varcoeff
        #print(z.size())
        return z #torch.mm(eps ,std)

    def Varianceanneal(self):
        if self.varcoeff <=1:
            self.varcoeff += self.varstep
            print(("Encoder variation scaling change to: %6.3f") %(self.varcoeff))




################################################# Decoder  #################################################

class RNNDecoder(nn.Module):
    """
    RNN decoder base class.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.worddropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)


    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        outputs = []
        '''
        input = input.type(torch.cuda.FloatTensor)
        mask = input.ge(0.3).type(torch.cuda.FloatTensor)
        input = input *mask
        input = input.type(torch.cuda.LongTensor)
        '''

        emb = self.embeddings(input)
        emb = self.worddropout(emb)
        #print(emb.size())

        #h0 = self._fix_enc_hidden(state.hidden[0])
        h0 =  state.hidden[0][0]
        #print(h0.size())
        h0_all = h0.repeat(emb.size()[0], 1,1)
        emb = torch.cat([emb, h0_all],2)
        #print(state.hidden[0].size())

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)
        outputs = self.dropout(rnn_output)

        # Result 
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0))

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        #for k in attns:
        #    attns[k] = torch.stack(attns[k])

        return outputs, state  #, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size + self.hidden_size



class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        #self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed) : #, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        #self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]



################################################# Generator #################################################
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size ):
        super(Generator, self).__init__()
        self.temperature = 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear(input)
        x = x.div(self.temperature)
        x = self.out(x)
        return x

################################################# NMT  #################################################
class VaeModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, model_opt, smultigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        super(VaeModel, self).__init__()
        z_size = model_opt.z_size
        hidden_size = model_opt.rnn_size
        self.encoder = encoder
        self.z2h = nn.Linear(z_size, hidden_size)
        self.z2c = nn.Linear(z_size, hidden_size)
        self.decoder = decoder
        self.n_layers = decoder.num_layers

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hid den (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        context, z,  mu, logvar = self.encoder(src, lengths)
        #enc_z = self.z2h(z)
        enc_z = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        cell_z = self.z2c(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        enc_state = self.decoder.init_decoder_state(context, (enc_z,cell_z))
        out, dec_state = self.decoder(tgt, context, enc_state)
        return out, dec_state, mu, logvar


