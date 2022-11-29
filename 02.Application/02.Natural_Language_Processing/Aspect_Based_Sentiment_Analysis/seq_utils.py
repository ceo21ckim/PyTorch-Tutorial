import torch 
import math 
import numpy as np 
'''
SimEval B-POS(NEG), I-POS(NEG), E-POS(NEG), O-POS(NEG), S-POS(NEG), EQ-POS(NEG) 형태로 입력됩니다.
O-POS(NEG), EQ-POS(NEG)는 아무 속성이 없는 단어를 의미하고,
B-POS(NEG), I-POS(NEG), E-POS(NEG), S-POS(NEG)는 단어의 속성을 나타냅니다.

original paper: https://arxiv.org/pdf/1910.00883.pdf

'''
def ot2bieos_ts(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        
        else:
            cur_pos, cur_sentiment = cur_ts_tag = cur_ts_tag.split('-')
            # cur_pos is 'T'
            if cur_pos != prev_pos:
                
                if i == n_tags -1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
                    
            else:
                if i == n_tags -1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos 
    return new_ts_sequence

def ot2bieos_ts_batch(ts_tag_seqs):
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bieos_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def tag2ts(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    begin, end = -1, -1
    
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        eles = ts_tag.split('-')
        
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        
        if sentiment != 'O':
            sentiments.append(sentiment)
        
        if pos == 'S':
            ts_sequence.append((i, i, sentiment))
            sentiments = []
            
        elif pos == 'B':
            beg = i 
            if len(sentiments) > 1:
                sentiments = [sentiments[-1]]
        
        elif pos == 'E':
            end = i
            
            if end > begin > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((begin, end, sentiment))
                sentiments = []
                begin, end = -1, -1
                
    return ts_sequence


def logsumexp(tensor, dim=-1, keepdim=True):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim).log())


def viterbi_decode(tag_sequence, transition_matrix, 
                   tag_observiations=None, allowed_start_transitions=None,
                   allowed_end_transitions=None):
    sequence_length, num_tags = list(tag_sequence.size())
    
    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None
    
    if has_start_end_restrictions:
        
        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
            
        if allowed_end_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)
        
        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix 
        
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        
        new_transition_matrix[-2, :] = allowed_start_transitions
        new_transition_matrix[-1, :] = -math.inf
        
        new_transition_matrix[:, -1] = allowed_end_transitions
        new_transition_matrix[:, -2] = -math.inf
        
        transition_matrix = new_transition_matrix
        
    if tag_observations:
        if len(tag_observations) != sequence_length:
            pass
        
    return transition_matrix
            
        
                        





















