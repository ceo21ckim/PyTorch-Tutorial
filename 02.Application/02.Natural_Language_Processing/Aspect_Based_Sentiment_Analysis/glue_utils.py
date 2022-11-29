import logging
import os, sys 

from seq_utils import * 

SMALL_POSITIVE_CONST = 1e-4

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        '''
        Inputs:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single 
                    sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            label: (Optional) string. The label of the example. This should be specified for train and dev examples.
        '''
        
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
class SeqInputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.evaluate_label_ids = evaluate_label_ids
        
class ABSAProcessor:
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines 
    
    ######
    def get_examples(self, data_dir, set_type='SimEval_train'):
        return self._create_examples(data_dir=data_dir, set_type=set_type)
    
    def get_labels(self):
        return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 
                 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG', 
                 'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
    
    def _create_examples(self, data_dir, set_type):
        examples = []
        file = os.path.join(data_dir, '%s.txt' % set_type)
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            sample_id = 0 
            for line in fp:
                '''
                sent_string: But the staff was so horrible to us.
                tag_string: But=O the=O staff=T-NEG was=O so=O horrible=O to=O us=O .=O

                words: [But, the, staff, was, so, horrible, to, us, .]
                tags: [O, O, S-NEG, O, O, O, O, O, O]
                '''
                sent_string, tag_string = line.strip().split('####')

                words, tags =[], []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        raise Exception('Invalid samples %s...' % tag_sting)

                    elif len(eles) == 2:
                        word, tag = eles

                    else:
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]

                    words.append(word)
                    tags.append(tag)

                tags = ot2bieos_ts(tags)

                guid = '%s-%s' % (set_type, sample_id)
                text_a = ' '.join(words)
                gold_ts = tag2ts(ts_tag_sequence=tag)
                for (_, _, s) in gold_ts:
                    if s == 'POS':
                        class_count[0] += 1

                    if s == 'NEG':
                        class_count[1] += 1

                    if s == 'NEU':
                        class_count[2] += 1

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1

            print('%s class count: %s' % (set_type, class_count))
            return examples


        
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break 
        
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        
        else:
            tokens_b.pop()

            
            
            
def convert_examples_to_seq_features(examples, label_list, tokenizer, 
                                     cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, 
                                     cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True):
    
    label_map = {label:i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1 
    examples_tokenized = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, labels_a = [], []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word, label in zip(words, example.label):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            if label != 'O':
                labels_a.extend([label] + ['EQ'] * (len(subwords) - 1))
            else:
                labels_a.extend(['O'] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            tid += len(subwords)
        
        assert tid == len(tokens_a)
        
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    
    max_seq_length += 2
    for ex_index, (tokens_a, labels_a, evaluate_label_ids) in enumerate(examples_tokenized):
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        
        
        # if cls_token_at_end
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        labels = ['O'] + labels
        evaluate_label_ids += 1
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        label_ids = [label_map[label] for label in labels]
        
        # if pad_on_left
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        
        label_ids = label_ids + ([0] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 10:
            logger.info('*** Example ***')
            logger.info('guid %s' % (example.guid))
            logger.info('tokens: %s' % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s " % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s " % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s " % " ".join([str(x) for x in label_ids]))
            logger.info("evaluate label ids: %s" % evaluate_label_ids)
        
        features.append(SeqInputFeatures(input_ids=input_ids, 
                                         input_mask=input_mask,
                                         segment_ids=segment_ids,
                                         label_ids=label_ids,
                                         evaluate_label_ids=evaluate_label_ids))
    print('maximal sequence length is %d' % (max_seq_length))
    return features 

    
    
def convert_examples_to_features(examples, label_list, max_seq_length, 
                                 tokenizer, cls_token='[CLS]', sep_token='[SEP]', 
                                 pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, 
                                 cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True):
    
    label_map = {label:i for i, label in enumerate(label_list)}
    
    features = []
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))
        
        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3) # [CLS], [SEP], [SEP]
        
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        '''
        tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids: 0    0   0    0    0      0    0   0    1  1  1  1  1   1
        
        '''
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]
        
        if ex_index < 10:
            logger.info('*** Example ***')
            logger.info('guid: %s' %(example.guid))
            logger.info('tokens: %s' % " ".join([str(x) for x in tokens]))
            logger.info('input_ids: %s' % " ".join([str(x) for x in input_ids]))
            logger.info('input_mask: %s' % " ".join([str(x) for x in input_mask]))
            logger.info('segment_ids: %s' % " ".join([str(x) for x in segment_ids]))
            logger.info('label: %s (id = %d)' % (example.label, label_id))
            
        feature.append(InputFeatures(input_ids=input_ids, 
                                     input_mask=input_mask,
                                     segment_ids=segment_ids, 
                                     label_id=label_id))
        
    return features 



def match_ts(gold_ts_sequence, pred_ts_sequence):
    '''
    Inputs:
        gold_ts_sequence: gold standard targeted sentiment sequence (ground truth)
        pred_ts_sequence: predicted targeted sentiment sequence
    '''
    
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
        
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count



def compute_metrics_absa(preds, labels, all_evaluate_label_ids):
    absa_label_vocab = {'O':0, 'EQ':1, 'B-POS':2, 'I-POS':3, 'E-POS':4, 'S-POS':5,
                        'B-NEG':6, 'I-NEG':7, 'E-NEG':8, 'S-NEG':9, 'B-NEU':10, 
                        'I-NEU':11, 'E-NEU':12, 'S-NEU':13}
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k 
        
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    n_samples = len(all_evaluate_label_ids)
    pred_y, gold_y = [], []
    class_count = np.zeros(3)
    
    for i in range(n_samples):
        evaluate_label_ids = all_evaluate_label_ids[i]
        pred_labels = preds[i][evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]
        
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)
        
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        
        for (_, _, s) in g_ts_sequence:
            if s == 'POS':
                class_count[0] += 1
            if s == 'NEG':
                class_count[1] += 1
            if s == 'NEU':
                class_count[2] += 1
    
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts +SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
        
    macro_f1 = ts_f1.mean()
    
    n_tp_total = sum(n_tp_ts)
    n_g_total = sum(n_gold_ts)
    print('class_count:', class_count)
    
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)
    scores = {'macro-f1':macro_f1, 'precision':micro_p, 'recall':micro_r, 'micro-f1':micro_f1}

    return scores 

# processors {
#     'total_rest': ABSAProcessor
# }

