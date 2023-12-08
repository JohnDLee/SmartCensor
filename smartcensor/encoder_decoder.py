import torch
import math, random, copy, sys, os

from layers import *
from utils import *
import bleu

# download required nltk packages
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('popular')
from nltk.tokenize import word_tokenize


torch.set_default_device('cpu') 
#torch.set_default_device('cuda') 

# The maximum length of any sentence, including <BOS> and <EOS>
max_len = 64

# Feed Forward Network module
class FFN(torch.nn.Module):
    def __init__(self, idims, hdims, odims, residual=True):
        super().__init__()
        self.lin1 = LinearLayer(idims, hdims)
        self.lin2 = LinearLayer(hdims, odims)
        self.residual = residual
        
    def forward(self, inp):
        hid = torch.relu(self.lin1(inp))
        out = self.lin2(hid)
        if self.residual:
            return inp + out
        else:
            return out

# Multi head self attention module
class MHSelfAttentionLayer(torch.nn.Module):
    def __init__(self, nheads, dims):
        super().__init__()
        self.heads = torch.nn.ModuleList([SelfAttentionLayer(dims) for h in range(nheads)])
        
    def forward(self, inp):
        return sum([h(inp) for h in self.heads]) / len(self.heads)

# A transformer is made up of a self attention layer followed up by a feed forward network
class TransformerBlock(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.att = MHSelfAttentionLayer(4, dims)
        self.ffn = FFN(dims, 4 * dims, dims)
    def forward(self, fencs):
        return self.ffn(self.att(fencs))
    
# encoder portion of the detoxifier
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims) 
        self.pos = Embedding(max_len, dims)    

        # 4 layer transformer encoder
        self.layers = torch.nn.Sequential(
            #torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4 * dims, dims),
            #torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4 * dims, dims),
            #torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4 * dims, dims),
            #torch.nn.Dropout(0.1),
            MHSelfAttentionLayer(4, dims),
            FFN(dims, 4 * dims, dims),
            #torch.nn.Dropout(0.1)
            )

    def forward(self, fnums):
        femb = self.emb(fnums)
        fpos = self.pos(torch.arange(len(fnums)))
        fencs = femb + fpos
        return self.layers(fencs)

# decoder portion of the detoxifier
class Decoder(torch.nn.Module):    
    def __init__(self, dims, vocab_size):
        super().__init__()
        self.dims = dims
        self.out = SoftmaxLayer(dims, vocab_size)              
        self.maskedAtt = MaskedSelfAttentionLayer(dims)        
        self.ffn = FFN(dims, 4 * dims, dims)                                   
        self.crossAtt = CrossAttentionLayer(dims)              

        self.eEmbed = Embedding(vocab_size, dims)             
        self.iEmbed = Embedding(max_len, dims)                 

    def start(self, fencs):
        return (self.maskedAtt.start(), 0, fencs)

    def step(self, state, enum):
        (prev_inputs, i, fencs) = state

        u = self.eEmbed(enum) + self.iEmbed(i)
        (new_inputs, g) = self.maskedAtt.step(prev_inputs, u)
        gprime = self.ffn(g)
        c = self.crossAtt(fencs, gprime)
        o = c + g
        
        return ((new_inputs, i+1, fencs), self.out(o))

    def forward(self, fencs, enums):
        u = self.eEmbed(enums) + self.iEmbed(torch.arange(len(enums)))
        g = self.ffn(self.maskedAtt(u))
        c = self.crossAtt(fencs, g)
        o = c + g
        return self.out(o)

class Model(torch.nn.Module):
    def __init__(self, fvocab, dims, evocab):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab
        
        self.encoder = Encoder(len(fvocab), dims)
        self.decoder = Decoder(dims, len(evocab))

        self.beam_k = 4

    def logprob(self, fwords, ewords):
        """Return the log-probability of a sentence pair.

        Arguments:
            fwords: source sentence (list of str)
            ewords: target sentence (list of str)

        Return:
            log-probability of ewords given fwords (scalar)"""

        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords])
        fencs = self.encoder(fnums)
        
        enums = torch.tensor([self.evocab.numberize(e) for e in ewords])
        ein = enums[:-1] # no <EOS>
        eout = enums[1:] # no <BOS>
        
        h = self.decoder(fencs, ein)
        logprobs = h[torch.arange(len(eout)), eout] # logprobs[i] = h[i,eout[i]]
        return logprobs.sum()

    def translate(self, fwords):
        """Translate a sentence using greedy search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords])
        fencs = self.encoder(fnums)
        state = self.decoder.start(fencs)
        ewords = []
        enum = self.evocab.numberize('<BOS>')
        for i in range(max_len-1):
            (state, elogprobs) = self.decoder.step(state, enum)
            enum = torch.argmax(elogprobs).item()
            eword = self.evocab.denumberize(enum)
            if eword == '<EOS>': break
            ewords.append(eword)
        return ewords

    def translate_beam(self, fwords):
        """Translate a sentence using beam search.

        Arguments:
            fwords: source sentence (list of str)

        Return:
            ewords: target sentence (list of str)
        """
        fnums = torch.tensor([self.fvocab.numberize(f) for f in fwords])
        fencs = self.encoder(fnums)
        ewords = []
        beam_state = [[[self.evocab.numberize('<BOS>')], self.decoder.start(fencs), 0]]
        for _ in range(max_len-1):
            new_beam_state = []
            for i in range(len(beam_state)):
                sentence, state, prob = beam_state[i]
                if (sentence[-1]) == self.evocab.numberize('<EOS>'): 
                    new_beam_state.append(beam_state[i])
                    continue
                enum = sentence[-1]
                (state, elogprobs) = self.decoder.step(state, enum)
                words, indices = torch.topk(elogprobs, self.beam_k)
                for j in range(self.beam_k):
                    new_beam_state.append([sentence + [indices[j].item()], state, prob + words[j]])
            beam_state = new_beam_state
            beam_state.sort(reverse=True, key=lambda x: x[2])
            beam_state = beam_state[:self.beam_k]
        #for sentence, state, prob in beam_state:
        #    print([self.evocab.denumberize(x) for x in sentence], prob)
        return [self.evocab.denumberize(x) for x in random.choice(beam_state)[0]]

def train(train_data, dev_data):
    fvocab = Vocab()
    evocab = Vocab()
    for fwords, ewords in train_data:
        fvocab |= fwords
        evocab |= ewords

    model = Model(fvocab, 256, evocab) 
    
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)

    best_dev_bleu = None
    for epoch in range(20):
        random.shuffle(train_data)

        train_loss = 0.
        train_ewords = 0
        for fwords, ewords in progress(train_data):
            loss = -model.logprob(fwords, ewords)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_ewords += len(ewords)-1 # includes EOS but not BOS

        dev_loss = 0.
        dev_ewords = 0
        dev_outputs = []
        print("Predicting the first 10 words:")
        for line_num, (fwords, ewords) in enumerate(progress(dev_data)):
            dev_loss -= model.logprob(fwords, ewords).item()
            dev_ewords += len(ewords)-1 # includes EOS but not BOS

            output = model.translate_beam(fwords)
            dev_outputs.append(output)
            if line_num < 10:
                print(str(line_num) + " original: " + fwords)
                print(str(line_num) + ": " + ' '.join(output), file=sys.stderr, flush=True)
        print("End predictions")

        dev_refs = [ewords for (_, ewords) in dev_data]
        dev_bleu = bleu.score(dev_outputs, dev_refs)
        if best_dev_bleu is None or dev_bleu > best_dev_bleu:
            best_model = copy.deepcopy(model)
            best_dev_bleu = dev_bleu
            torch.save(model, f"../model_{epoch+1}.pt")

        print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)} dev_bleu={dev_bleu}', file=sys.stderr, flush=True)

    return best_model

if __name__ == "__main__":
    train_data = []
    dev_data = []
    with open("data/train.tsv", "r") as f:
         f.readline()
         for line in f:
             toxic, nontoxic = line.strip().split("\t")
             train_data.append((["<BOS>"] + word_tokenize(toxic) + ["<EOS>"], ["<BOS>"] + word_tokenize(nontoxic) + ["<EOS>"]))
    dev_train_split = 19744-19744//5
    dev_data = train_data[dev_train_split:]
    train_data = train_data[:dev_train_split]
    model = train(train_data, dev_data)
    
    #model = torch.load(os.path.join(out_dir, 'mymodel.pt'))
    
    #print(f'[done] test_bleu={bleu.score(test_outputs, test_refs)}')
