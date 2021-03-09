# coding=utf-8

""" C2C-GenDA fine-tuning script with OpenAI GPT model. """

import argparse
import os
import random
import logging
import json
import math
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from transformers import (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, WEIGHTS_NAME, CONFIG_NAME,
                          WarmupLinearSchedule)
import pickle


def kl_anneal_function(anneal_function, step, k, x0, weight):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0)))) * weight
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == 'constant':
        return weight
    elif anneal_function == 'zero':
        return weight if step >= x0 else 0
    elif anneal_function == 'updown':
        return ((step // x0) % 2) * weight


def load_da_dataset(args, dataset_path, trg_file=True, unk_token_old='<UNK>', unk_token_new="<|endoftext|>"):
    """ Load the cluster-to-cluster pairs and output the list:
            if trg_file=True:
                [[[source_seq_0, ..., source_seq_n, rank_i, target_seq_i] for ith_pair] for each cluster]
            else:
                [[[source_seq_0, ..., source_seq_n, rank_i] for ith_pair] for each cluster]
    """
    src_path = dataset_path.split('.')
    src_path[-2] += '_src'
    src_path = '.'.join(src_path)
    s_f = open(src_path, 'r', encoding='utf_8')
    s_l = s_f.readline()

    if trg_file:
        trg_path = dataset_path.split('.')
        trg_path[-2] += '_tgt'
        trg_path = '.'.join(trg_path)
        t_f = open(trg_path, 'r', encoding='utf_8')
        t_l = t_f.readline()

    output = []
    while s_l:
        cluster = []
        s_l = s_l.replace(unk_token_old, unk_token_new)
        s_sentences = s_l.split('\t')
        if trg_file:
            t_l = t_l.replace(unk_token_old, unk_token_new)
            t_sentences = t_l.split('\t')
            for idx, t_sentence in enumerate(t_sentences):
                pair_sentences = s_sentences + ['<'+str(idx)+'>', t_sentence]
                cluster.append(pair_sentences)
        else:
            for idx in range(args.target_cluster_num_threshold):
                pair_sentences = s_sentences + ['<' + str(idx) + '>']
                cluster.append(pair_sentences)
        output.append(cluster)
        s_l = s_f.readline()
        if trg_file:
            t_l = t_f.readline()
    if trg_file:
        assert s_l == t_l
    return output


def pre_process_datasets(args, encoded_dataset, max_len, bos_token_id, eos_token_id, lm_required=True):
    """ Pre-processing and output tuple(input_ids, lm_labels, len_sources) if lm_required else (input_ids, len_sources):
            input_ids[cluster, rank, :] = [bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank +
                                        [eos_token_id, bos_token_id] + target_sentence + [eos_token_id, eos_token_id]
            lm_labels[cluster, rank, :] = [-1] * len([bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank) +
                                        [eos_token_id, bos_token_id] + target_sentence + [eos_token_id, eos_token_id]
            len_sources[cluster] = len([bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank +
                                        [eos_token_id, bos_token_id])
    """
    n_batch = len(encoded_dataset)
    input_ids = np.zeros((n_batch, args.target_cluster_num_threshold, max_len), dtype=np.int64)
    len_sources = np.zeros((n_batch), dtype=np.int64)
    if lm_required:
        lm_labels = np.full((n_batch, args.target_cluster_num_threshold, max_len), fill_value=-1, dtype=np.int64)

    special_tokens_length = 7  # 3 * [bos_token_id] + 4 * [eos_token_id]
    input_len = 0
    for idx_c, cluster in enumerate(encoded_dataset):
        for idx_p, pair in enumerate(cluster):
            rank = pair[-2]
            target_sentence = pair[-1]
            idx_s = 0
            source_sentences = pair[idx_s]
            while idx_s + 2 < len(pair) - 1 and len(source_sentences) + 2 + len(pair[idx_s + 1]) + len(rank) + len(
                    target_sentence) + special_tokens_length <= max_len:
                idx_s += 1
                source_sentences += [eos_token_id, bos_token_id] + pair[idx_s]

            pair_with_st = [bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank + \
                           [eos_token_id, bos_token_id] + target_sentence + [eos_token_id, eos_token_id]

            len_pair = len(pair_with_st)
            input_len = max(input_len, len_pair)
            input_ids[idx_c, idx_p, :len_pair] = pair_with_st
            len_source = len([bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank +
                             [eos_token_id, bos_token_id])

            if len_sources[idx_c]:
                assert len_sources[idx_c] == len_source
            len_sources[idx_c] = len_source

            if lm_required:
                label_pair_with_st = [-1] * len([bos_token_id] + source_sentences +
                                                [eos_token_id, bos_token_id] + rank) + \
                                     [eos_token_id, bos_token_id] + target_sentence + [eos_token_id, eos_token_id]

                assert len(pair_with_st) == len(label_pair_with_st)
                lm_labels[idx_c, idx_p, :len_pair] = label_pair_with_st

    all_inputs = (input_ids, lm_labels, len_sources) if lm_required else (input_ids, len_sources)
    assert torch.tensor(input_ids).narrow(2, input_len, max_len-input_len).nonzero().shape[0] == 0
    tensor_dataset = tuple(torch.tensor(t).narrow(2, 0, input_len) for t in all_inputs[:-1]) + \
                     (torch.tensor(all_inputs[-1]).unsqueeze(-1),)
    return tensor_dataset


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, rank_tokens_ids=None, slots_tokens_ids=None,
                    stop_early_id=None, greed=False, empty_accept=True, argmax_slots=False,
                    intra_attn=False, intra_attention_weight=None,
                    num_samples=1, temperature=1, top_k=0, top_p=0.0, device=torch.device("cpu")):
    context = torch.tensor(context, dtype=torch.long, device=device)
    if len(context.size()) == 1:
        context = context.unsqueeze(0)
    context = context.repeat(num_samples, 1)
    len_source = torch.tensor(context.size(-1)).unsqueeze(0)

    generated = context
    with torch.no_grad():
        stopped = [False] * generated.size(0)
        for t in range(length):
            inputs = {'input_ids': generated, 'intra_attention': intra_attn, 'len_source': len_source,
                      'intra_attention_weight': intra_attention_weight}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            next_tokens = []
            for idx_sample in range(generated.size(0)):
                next_token_logits = outputs[idx_sample, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                filtered_probs = F.softmax(filtered_logits, dim=-1)
                # filter rank tokens
                if filtered_probs[rank_tokens_ids].sum() == 1:
                    filtered_probs[stop_early_id] = filtered_probs[rank_tokens_ids].sum()
                filtered_probs[rank_tokens_ids] = 0
                # early stop
                if not empty_accept and (t == 0 or (t == 1 and generated[idx_sample][-1] == stop_early_id)):
                    filtered_probs[stop_early_id] = 0
                if stopped[idx_sample]:
                    filtered_probs.zero_()
                    filtered_probs[stop_early_id] = 1
                # sample next token
                argmax_token = torch.argmax(filtered_probs).unsqueeze(-1)
                if argmax_slots and argmax_token[0] in slots_tokens_ids:
                    print('predict slot token {}, p={}'.format(argmax_token[0], filtered_probs[argmax_token]))
                    next_token = argmax_token
                elif greed:
                    next_token = argmax_token
                else:
                    next_token = torch.multinomial(filtered_probs, num_samples=1)
                next_tokens.append(next_token.unsqueeze(0))

            next_tokens = torch.cat(next_tokens, dim=0)
            generated = torch.cat((generated, next_tokens), dim=1)

            for idx_sample in range(generated.size(0)):
                if stop_early_id and not stopped[idx_sample] and generated[idx_sample][-1] == stop_early_id and \
                        generated[idx_sample][-2] == stop_early_id:
                    stopped[idx_sample] = True
            if all(stopped) or generated.size(1) > 1024:
                break

    return generated


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='pretrained model name')

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_gen", action='store_true', help="Whether to generation new target sentence")

    parser.add_argument("--log_output_path", default='./log.txt', type=str,
                        help="The output directory where the log files will be written.")
    parser.add_argument("--model_output_dir", default='./model', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--gen_output_dir", default='/data/AugmentedData/gen/', type=str,
                        help="The output directory where the augmented data will be saved.")

    parser.add_argument('--train_dataset', type=str, default='./data/AugmentedData/atis_train_clustered_train.txt')
    parser.add_argument('--gen_dataset', type=str, default='./data/AugmentedData/atis_train_clustered_reserve.txt')
    parser.add_argument('--slots_tokens_path', type=str,
                        default='./data/Cluster/atis_train_slot2entity_dictionary.json')
    parser.add_argument('--original_data_path', type=str, default='./data/Cluster/atis_train_all_sentences.txt')
    parser.add_argument("--target_cluster_num_threshold", type=int, default=10, help="target_cluster_threshold")
    parser.add_argument("--unknown_token", type=str, default="<UNK>", help="unknown_token in the dataset")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help="the number of the cluster per training batch")
    parser.add_argument('--train_target_size', type=int, default=5,
                        help="the number of the target utterances per training batch, "
                             "which should be less than train_batch_size * target_cluster_num_threshold")

    parser.add_argument("--intra_kl_loss", action='store_true',
                        help='add KL loss among the target utterances')
    parser.add_argument('--intra_kl_loss_weight', type=float, default=1)
    parser.add_argument('--intra_kl_anneal_func', type=str, default='constant',
                        choices=["logistic", "linear", 'constant', 'zero', 'updown', ])
    parser.add_argument('--intra_kl_k', type=float, default=0.0025)
    parser.add_argument('--intra_kl_x0', type=int, default=2500)

    parser.add_argument("--intra_attention", action='store_true',
                        help='add attention among the target utterances')
    parser.add_argument('--intra_attention_weight', type=float, default=0.01)

    parser.add_argument("--wo_pretrained", action='store_true',
                        help='training with the random initialized model')
    parser.add_argument('--wo_pretrained_layer', type=int, default=2)

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_save_per_epoch', type=int, default=1)
    parser.add_argument('--n_save_epochs', type=int, default=1)
    parser.add_argument('--n_valid', type=int, default=374)

    parser.add_argument('--gen_length', type=int, default=40,
                        help="max length of the generation utterances")
    parser.add_argument('--gen_mode', type=str, default='sample', choices=["sample", "greed"],
                        help="generate each token in greed or sample mode")
    parser.add_argument("--gen_accept_empty", action='store_true',
                        help="accept empty utterances generation")
    parser.add_argument("--gen_stop_early", action='store_true',
                        help="stop generation if two eos token are generated continuously")
    parser.add_argument("--gen_argmax_slots", action='store_true',
                        help="generate each slot in greed or sample mode")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print('args: ', args)

    # Get ready...
    logging.basicConfig(filename=args.log_output_path,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('\nargs: {}'.format(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    # Load tokenizer and model
    with open(args.slots_tokens_path, 'r') as slots_file:
        slots_dict = json.load(slots_file)
        slots_tokens = list(slots_dict.keys())
    rank_tokens = ['<'+str(i)+'>' for i in range(args.target_cluster_num_threshold)]
    special_tokens = slots_tokens + rank_tokens
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(special_tokens)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    slots_tokens_ids = tokenizer.convert_tokens_to_ids(slots_tokens)
    rank_tokens_ids = tokenizer.convert_tokens_to_ids(rank_tokens)
    logger.info("special_tokens: {}".format(special_tokens))
    logger.info("special_tokens_ids: {}".format(special_tokens_ids))
    logger.info("special_tokens: {}".format(tokenizer.special_tokens_map))
    bos_token, eos_token, unk_token = tokenizer.special_tokens_map['bos_token'], \
                                      tokenizer.special_tokens_map['eos_token'], \
                                      tokenizer.special_tokens_map['unk_token']
    bos_token_id, eos_token_id, unk_token_id = tokenizer.convert_tokens_to_ids([bos_token, eos_token, unk_token])
    if args.wo_pretrained:
        config = GPT2Config.from_pretrained(args.model_name, n_layer=args.wo_pretrained_layer)
        model = GPT2LMHeadModel(config)
        logger.info("GPT2 model without pretrained: {}".format(model))
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Load and encode the datasets
    logger.info("Loading and Encoding dataset...")

    def tokenize_and_encode(data):
        """ Tokenize and encode a nested object """
        encoded_data = []
        for cluster in tqdm(data, desc='Tokenize and Encode'):
            encoded_data.append([[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) for s in pair]
                                 for pair in cluster])
        return encoded_data

    encoded_datasets = {}
    if args.do_train:
        encoded_path = args.train_dataset.split('.')
        encoded_path[-2] += '_encoded'
        encoded_path = '.'.join(encoded_path)
        if os.path.exists(encoded_path):
            encoded_datasets.update(pickle.load(open(encoded_path, "rb")))
        else:
            train_dataset = load_da_dataset(args, args.train_dataset,
                                            unk_token_old=args.unknown_token, unk_token_new=unk_token)
            encoded_datasets['train'] = tokenize_and_encode(train_dataset)
            with open(encoded_path, "wb") as f:
                pickle.dump(encoded_datasets, f)
    if args.do_gen:
        encoded_path = args.gen_dataset.split('.')
        encoded_path[-2] += '_gen_encoded'
        encoded_path = '.'.join(encoded_path)
        if os.path.exists(encoded_path):
            encoded_datasets.update(pickle.load(open(encoded_path, "rb")))
        else:
            gen_dataset = load_da_dataset(args, args.gen_dataset, trg_file=False,
                                          unk_token_old=args.unknown_token, unk_token_new=unk_token)
            encoded_datasets['gen'] = tokenize_and_encode(gen_dataset)
            with open(encoded_path, "wb") as f:
                pickle.dump(encoded_datasets, f)

    log_data_type = 'train' if args.do_train else 'gen'
    rank_idx = -1 if log_data_type == 'gen' else -2
    for i in range(2):
        for j in range(min(5, len(encoded_datasets[log_data_type][i]))):
            logger.info("\n*****Examples*****")
            logger.info('Source tokens: {}'.format(
                [tokenizer.convert_ids_to_tokens(d) for d in encoded_datasets[log_data_type][i][j][:rank_idx]]))
            logger.info('Source ids: {}'.format(encoded_datasets[log_data_type][i][j][:rank_idx]))
            logger.info('Rank tokens: {}'.format(
                [tokenizer.convert_ids_to_tokens(d) for d in encoded_datasets[log_data_type][i][j][rank_idx]]))
            logger.info('Rank ids: {}'.format(encoded_datasets[log_data_type][i][j][rank_idx]))
            if log_data_type != 'gen':
                logger.info('Target tokens: {}'.format(
                    [tokenizer.convert_ids_to_tokens(d) for d in encoded_datasets[log_data_type][i][j][-1]]))
                logger.info('Target ids: {}'.format(encoded_datasets[log_data_type][i][j][-1]))
            logger.info('')

    # Train
    if args.do_train:

        # Prepare inputs tensors and dataloaders
        encoded_datasets_train = encoded_datasets['train']
        train_tensor_dataset = pre_process_datasets(args, encoded_datasets_train, model.config.n_positions, bos_token_id, eos_token_id)

        train_data = TensorDataset(*train_tensor_dataset)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Prepare optimizer
        step_num_per_batch = math.ceil(args.train_batch_size * args.target_cluster_num_threshold / args.train_target_size)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) * step_num_per_batch // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) * step_num_per_batch // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        def save_model(model, tokenizer, time):
            model_output_dir = os.path.join(args.model_output_dir, str(time))
            version = 1
            while os.path.exists(model_output_dir):
                model_output_dir = os.path.join(args.model_output_dir, str(time)+'.'+str(version))
                version += 1
            os.makedirs(model_output_dir)
            logger.info("\nSaving the model to {}\n".format(os.path.join(model_output_dir)))

            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(model_output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(model_output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(model_output_dir)

        # Let's train
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.train()

        update_step, total_step, exp_average_loss = 0, 0, None
        for idx_epochs in trange(int(args.num_train_epochs), desc="Epoch"):
            total_step = 0
            num_save_checkpoint = len(train_dataloader) // args.n_save_per_epoch
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            optimizer.zero_grad()
            for idx_batch, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels, len_sources = batch
                len_sources = len_sources.expand(-1, input_ids.shape[1]).view(input_ids.shape[0] * input_ids.shape[1])
                input_ids = input_ids.view(input_ids.shape[0] * input_ids.shape[1], input_ids.shape[2])
                lm_labels = lm_labels.view(lm_labels.shape[0] * lm_labels.shape[1], lm_labels.shape[2])

                batch_size = input_ids.shape[0]
                random_indices = torch.tensor(random.sample(range(batch_size), batch_size)).to(device)
                for idx_step in range(step_num_per_batch):
                    idx_begin = idx_step * args.train_target_size
                    idx_end = min(idx_begin + args.train_target_size, input_ids.shape[0])
                    step_size = idx_end-idx_begin
                    input_ids_step = input_ids.index_select(0, random_indices[idx_begin:idx_end])
                    lm_labels_step = lm_labels.index_select(0, random_indices[idx_begin:idx_end])
                    len_sources_step = len_sources.index_select(0, random_indices[idx_begin:idx_end])

                    assert len(len_sources_step.unique()) == 1
                    logits = model(input_ids_step, intra_attention=args.intra_attention, len_source=len_sources_step,
                                   intra_attention_weight=args.intra_attention_weight)

                    # calculate the nll loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels_step[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # calculate the kl loss
                    if args.intra_kl_loss:
                        kl_loss_fun = torch.nn.KLDivLoss(reduce=False)
                        kl_loss = 0

                        # calculate loss within the current step
                        for idx_sample in range(1, step_size):
                            indices = torch.tensor(list(range(idx_sample, step_size)) +
                                                   list(range(0, idx_sample))).to(device)
                            disper_logits = logits.detach().index_select(0, indices)
                            log_q = F.log_softmax(logits, dim=-1)
                            p = F.softmax(disper_logits, dim=-1)
                            kl_loss_step = kl_loss_fun(log_q, p)
                            kl_weight = kl_anneal_function(args.intra_kl_anneal_func, update_step,
                                                           args.intra_kl_k, args.intra_kl_x0, args.intra_kl_loss_weight)
                            kl_loss += -kl_loss_step.sum(dim=-1).mean() * kl_weight

                        # calculate loss outside the current step
                        indices_outside = torch.cat((random_indices[:idx_begin], random_indices[idx_end:]))
                        input_ids_outside = input_ids.index_select(0, indices_outside)
                        len_sources_outside = len_sources.index_select(0, indices_outside)
                        with torch.no_grad():
                            assert len(len_sources_outside.unique()) == 1
                            logits_outside = model(input_ids_outside, intra_attention=args.intra_attention,
                                                   len_source=len_sources_outside,
                                                   intra_attention_weight=args.intra_attention_weight)
                        for idx_sample in range(batch_size - step_size):
                            indices = torch.tensor([idx_sample for _ in range(step_size)]).to(device)
                            disper_logits = logits_outside.detach().index_select(0, indices)
                            log_q = F.log_softmax(logits, dim=-1)
                            p = F.softmax(disper_logits, dim=-1)
                            kl_loss_step = kl_loss_fun(log_q, p)
                            kl_weight = kl_anneal_function(args.intra_kl_anneal_func, update_step,
                                                           args.intra_kl_k, args.intra_kl_x0, args.intra_kl_loss_weight)
                            kl_loss += -kl_loss_step.sum(dim=-1).mean() * kl_weight

                        kl_loss = kl_loss / (batch_size - 1)
                        loss += kl_loss
                    else:
                        kl_loss = torch.zeros_like(loss)
                    loss = loss / args.gradient_accumulation_steps

                    # backward, step and report
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if (total_step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                        update_step += 1

                    exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                    total_step += 1
                    tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])
                    kl_weight = kl_anneal_function(args.intra_kl_anneal_func, update_step,
                                                   args.intra_kl_k, args.intra_kl_x0, args.intra_kl_loss_weight)
                    logger.info(
                        "Epoch: {}, Step: {}, Training loss: {:.2e} current loss: {:.2e} "
                        "nll loss: {:.2e} kl loss: {:.2e} lr: {:.2e} update_step: {:.2e} kl weight {:.2e}\n".format(
                            idx_epochs, total_step, exp_average_loss, loss.item(),
                            loss.item() - kl_loss.item(), kl_loss.item(),
                            scheduler.get_lr()[0], update_step, kl_weight))

                if (idx_epochs + 1) % args.n_save_epochs == 0 and (idx_batch + 1) % num_save_checkpoint == 0:
                    save_model(model, tokenizer,
                               'epoch_'+str(idx_epochs)+'_batch_'+str(idx_batch)+'_step_'+str(total_step))

    # Generation
    if args.do_gen:
        encoded_datasets_gen = encoded_datasets['gen']

        if not os.path.exists(args.gen_output_dir):
            os.mkdir(args.gen_output_dir)
        gen_file_name = os.path.split(args.gen_dataset)[-1].split('.')
        gen_file_name[-2] += '_gen'
        gen_file_name = '.'.join(gen_file_name)
        gen_file_path = os.path.join(args.gen_output_dir, gen_file_name)
        gen_file = open(gen_file_path, 'w')

        for cluster in tqdm(encoded_datasets_gen, desc='Generate target utterances', total=len(encoded_datasets_gen)):
            pairs_with_st = []
            for pair in cluster:
                rank = pair[-1]
                source_sentences = pair[0]
                for source_sentence in pair[1:-1]:
                    source_sentences += [eos_token_id, bos_token_id] + source_sentence
                pair_with_st = [bos_token_id] + source_sentences + [eos_token_id, bos_token_id] + rank + \
                               [eos_token_id, bos_token_id]
                pairs_with_st.append(pair_with_st)

            # Generation without intra_attention
            if not args.intra_attention:
                for pair_with_st in pairs_with_st:
                    out = sample_sequence(
                        model=model,
                        context=pair_with_st,
                        length=args.gen_length,
                        rank_tokens_ids=rank_tokens_ids,
                        slots_tokens_ids=slots_tokens_ids,
                        stop_early_id=eos_token_id if args.gen_stop_early else None,
                        greed=True if args.gen_mode == 'greed' else False,
                        empty_accept=True if args.gen_accept_empty else False,
                        argmax_slots=args.gen_argmax_slots,
                        intra_attn=args.intra_attention,
                        intra_attention_weight=args.intra_attention_weight,
                        device=device
                    )
                    if args.gen_stop_early:
                        out = out[0, len(pair_with_st):-2].tolist()
                    else:
                        out = out[0].tolist()
                    text = tokenizer.decode(out)
                    gen_file.write(text.replace(unk_token, args.unknown_token).replace('\n', ' ')+'\n')
                    gen_file.flush()
            # Generation with intra_attention
            else:
                out = sample_sequence(
                    model=model,
                    context=pairs_with_st,
                    length=args.gen_length,
                    rank_tokens_ids=rank_tokens_ids,
                    slots_tokens_ids=slots_tokens_ids,
                    stop_early_id=eos_token_id if args.gen_stop_early else None,
                    greed=True if args.gen_mode == 'greed' else False,
                    empty_accept=True if args.gen_accept_empty else False,
                    argmax_slots=args.gen_argmax_slots,
                    intra_attn=args.intra_attention,
                    intra_attention_weight=args.intra_attention_weight,
                    device=device
                )
                for idx_out in range(out.size(0)):
                    text = out[idx_out].tolist()
                    if args.gen_stop_early:
                        idx_end = -2
                        while text[idx_end] == eos_token_id:
                            idx_end -= 1
                        text = text[len(pairs_with_st[0]):idx_end+1]
                    text = tokenizer.decode(text)
                    gen_file.write(text.replace(unk_token, args.unknown_token).replace('\n', ' ')+'\n')
                    gen_file.flush()
        gen_file.close()

        # Combine the original utterances with the generated utterances to the whole augmented utterances
        augmented_file_name = os.path.split(args.gen_dataset)[-1].split('clustered')[:-1]
        augmented_file_name[-1] += 'augmented'
        augmented_file_name = 'clustered'.join(augmented_file_name)
        augmented_file_path = os.path.join(args.gen_output_dir, augmented_file_name)
        with open(gen_file_path, 'r') as gen_file, open(args.original_data_path, 'r') as origin_file, \
                open(augmented_file_path, 'w') as augmented_file:
            for line in origin_file:
                augmented_file.write(line)
            for line in gen_file:
                augmented_file.write(line)


if __name__ == '__main__':
    main()
