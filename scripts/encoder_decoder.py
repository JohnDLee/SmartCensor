# Trains, Tests, or Demos the Encoder Decoder architecture. Has a toxification feature.


import torch
import math, random, copy, sys, os
import argparse


from detoxify.detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util

from smartcensor.data import JigsawData, JigsawPath
from smartcensor.layers import *
from smartcensor.utils import *
import smartcensor.bleu as bleu

# download required nltk packages
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('popular', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


# torch.set_default_device('cpu') 
torch.set_default_device('cuda') 

# The maximum length of any sentence, including <BOS> and <EOS>
max_len = 64

class FFN(torch.nn.Module):
    """Feed Forward Network module"""
    
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


class MHSelfAttentionLayer(torch.nn.Module):
    """Multi head self attention module"""
    def __init__(self, nheads, dims):
        super().__init__()
        self.heads = torch.nn.ModuleList([SelfAttentionLayer(dims) for h in range(nheads)])
        
    def forward(self, inp):
        return sum([h(inp) for h in self.heads]) / len(self.heads)


class TransformerBlock(torch.nn.Module):
    """A transformer is made up of a self attention layer followed up by a feed forward network"""
    def __init__(self, dims):
        super().__init__()
        self.att = MHSelfAttentionLayer(4, dims)
        self.ffn = FFN(dims, 4 * dims, dims)
    def forward(self, fencs):
        return self.ffn(self.att(fencs))
    
class Encoder(torch.nn.Module):
    """ encoder portion of the detoxifier """
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

class Decoder(torch.nn.Module):    
    """decoder portion of the detoxifier"""
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
    def __init__(self, fvocab, dims, evocab, beam_k = 4):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.fvocab = fvocab
        self.evocab = evocab
        
        self.encoder = Encoder(len(fvocab), dims)
        self.decoder = Decoder(dims, len(evocab))

        self.beam_k = beam_k

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

def train(train_data, dev_data, detokenizer, num_epochs = 10, save_dir='models_detoxifier' ):
    """Training Routine"""
    
    fvocab = Vocab()
    evocab = Vocab()
    for fwords, ewords in train_data:
        fvocab |= fwords
        evocab |= ewords
    
    # Model
    model = Model(fvocab, 256, evocab) 
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    best_dev_bleu = None
    for epoch in range(num_epochs):
        # shuffle training data
        random.shuffle(train_data)
        model.train()
        train_loss = 0.
        train_ewords = 0
        for fwords, ewords in progress(train_data, desc=f'Training Epoch {epoch+1}', total = len(train_data)):
            # loss is negative log prob of words
            loss = -model.logprob(fwords, ewords)
            # backpropogate
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_ewords += len(ewords)-1 # includes EOS but not BOS

        dev_loss = 0.
        dev_ewords = 0
        dev_outputs = []
        model.eval()
        print("Predicting the first 10 sentences:")
        for line_num, (fwords, ewords) in enumerate(progress(dev_data, desc=f'Validating Epoch {epoch+1}', total = len(dev_data), miniters=10)):
            dev_loss -= model.logprob(fwords, ewords).item()
            dev_ewords += len(ewords)-1 # includes EOS but not BOS

            # translate using beam search
            output = model.translate_beam(fwords)
            dev_outputs.append(output)
            if line_num < 10:
                # print(str(line_num) + " original : " + ' '.join(fwords[1:-1]))
                # print(str(line_num) + " predicted: " + ' '.join(output[1:-1]), file=sys.stderr, flush=True)
                # print()
                print(str(line_num) + " original : " + detokenizer.detokenize(fwords[1:-1]))
                print(str(line_num) + " predicted: " + detokenizer.detokenize(output[1:-1]), file=sys.stderr, flush=True)
                print()
        print("End predictions")

        dev_refs = [ewords for (_, ewords) in dev_data]
        dev_bleu = bleu.score(dev_outputs, dev_refs)
        if best_dev_bleu is None or dev_bleu > best_dev_bleu:
            best_model = copy.deepcopy(model)
            best_dev_bleu = dev_bleu
            torch.save(model, f"{save_dir}/model_{epoch+1}.pt")

        print(f'[{epoch+1}] train_loss={train_loss} train_ppl={math.exp(train_loss/train_ewords)} dev_ppl={math.exp(dev_loss/dev_ewords)} dev_bleu={dev_bleu}', file=sys.stderr, flush=True)

    return best_model

def detoxify(input_sent, model, d):
    """Detoxify a sentence using our model"""
    sent = sent_tokenize(input_sent)
    outputs = []
    for input_word in sent:
        word = word_tokenize(input_word)[:max_len - 2]
        inps = ["<BOS>"] + word + ["<EOS>"]
        output = model.translate_beam(inps)
        outputs.append(d.detokenize(output[1:-1]))
    return ' '.join(outputs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train and Encoder-Decoder for detoxification")
    subparsers = parser.add_subparsers(dest='mode')
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-data', default='data/train.tsv', help='training data')
    train_parser.add_argument('-toxifier', action='store_true', default=False, help='to train a toxifier')
    train_parser.add_argument('-dev', action='store_true', help='dev mode')
    
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('-model', help='best model to use')
    test_parser.add_argument('-eval_mode', choices=['all', 'toxic', 'nontoxic'], default='all', help='whether to evaluate on all, toxic, or nontoxic values')
    
    demo_parser = subparsers.add_parser('demo')
    demo_parser.add_argument('-model', help='best model to use')
    
    args = parser.parse_args()
    
    d = TreebankWordDetokenizer()
    if args.mode == 'train':        
        
        if not args.toxifier:
            
            train_data = []
            dev_data = []
            with open(args.data, "r") as f:
                f.readline()
                for line in f:
                    toxic, nontoxic = line.strip().split("\t")
                    tokenized = (["<BOS>"] + word_tokenize(toxic) + ["<EOS>"], ["<BOS>"] + word_tokenize(nontoxic) + ["<EOS>"])
                    train_data.append(tokenized)
            
            #! for dev
            if args.dev:
                train_data = train_data[:100]
                
            dev_train_split = len(train_data) - len(train_data)//5
            dev_data = train_data[dev_train_split:]
            train_data = train_data[:dev_train_split]
            model = train(train_data, dev_data, d, save_dir='models_detoxifier')
            
            
        elif args.toxifier:
            #train toxifier
            train_data = []
            dev_data = []
            with open(args.data, "r") as f:
                f.readline()
                for line in f:
                    toxic, nontoxic = line.strip().split("\t")
                    tokenized = (["<BOS>"] + word_tokenize(nontoxic) + ["<EOS>"], ["<BOS>"] + word_tokenize(toxic) + ["<EOS>"],)
                    train_data.append(tokenized)
            
            #! for dev
            if args.dev:
                train_data = train_data[:100]
                
            dev_train_split = len(train_data) - len(train_data)//5
            dev_data = train_data[dev_train_split:]
            train_data = train_data[:dev_train_split]
            model = train(train_data, dev_data, d, save_dir='models_toxifier')
    
    
    elif args.mode == 'test':
        """ Only evaluates detoxifier"""
        model = torch.load(args.model)
        model.eval()
        
        ## Load Testing Data
        baseline_data = JigsawData(JigsawPath.test)
        zipped_data = baseline_data.get_zipped_data()
        ## Load Toxic Bert
        toxic_metric = Detoxify('original')
        ## Load Sentence embedder
        sentence_emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Models Loaded.")
        keys = toxic_metric.predict('')
        reductions = {k: [] for k in keys.keys()}
        sim = []
        
        for id, comment, labels in progress(zipped_data, desc='Cleaning Strings'):
            # for every toxic word, sub it for empty
            if args.eval_mode == 'toxic' and sum(labels) == 0:
                continue
            elif args.eval_mode == 'nontoxic' and sum(labels) > 0:
                continue
            
            clean_string = detoxify(comment, model, d)
            ## Oberve Toxicity
            results = toxic_metric.predict([comment, clean_string])
            for k in reductions.keys():
                reductions[k].append(results[k][1] - results[k][0])
            
            
            ## Observe Sentence Cosine Sim
            embedding_toxic = sentence_emb.encode(comment, convert_to_tensor=True)
            embedding_clean = sentence_emb.encode(clean_string, convert_to_tensor=True)
            sim.append(util.pytorch_cos_sim(embedding_toxic, embedding_clean).item())
            
        ## Average Toxicity Reduction
        print("Average Reductions in Toxicity:")
        for k, val in reductions.items():
            print(f'{k:20} | {sum(val)/len(val):.4f}')
        
        print(f"Average Cosine Similarity: {sum(sim)/len(sim):.4f}")
    
    elif args.mode == 'demo':
        model = torch.load(args.model)
        print("Model Loaded:")
        model.eval()
        for line in sys.stdin:
            print("Predicted: " + detoxify(line.rstrip(), model, d), file=sys.stderr, flush=True)
        #print(f'[done] test_bleu={bleu.score(test_outputs, test_refs)}')
