import math
import os
import random
import re
import pickle
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher
from IPython import embed
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class TM_Generation_1step(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(TM_Generation_1step, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.model_conf = model_conf
        self.device = device

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']
        self.decoder_lr = model_conf['decoder_lr']

        if self.decoder_lr == -1:
            self.decoder_lr = self.lr

        self.reg = model_conf['reg']
        self.demo = model_conf['demo']
        self.beam_size = model_conf['beam_size']

        self.decoder_num_layers = model_conf['decoder_num_layers']
        self.decoder_num_heads = model_conf['decoder_num_heads']
        self.num_template_vocab = len(dataset.templatetoken2idx)
        # self.num_net_vocab = len(dataset.netvocab2netidx)
        # self.num_op_vocab = len(self.dataset.operator2idx)

        self.pretrained_path = model_conf['pretrained_path']
        self.model_weight_path = model_conf['model_weight_path']
        self.bert_conf = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_max_len = model_conf['bert_max_len']
        
        self.do_mask_imq = model_conf.get('do_mask_imq', False)
        self.imq_mask_ratio = model_conf.get('imq_mask_ratio', 0.1)
        
        self.decode_only_input_token = model_conf['decode_only_input_token']
        self.use_category = model_conf['use_category']
        self.lr_schedule = model_conf['lr_schedule']
        self.warmup_rate = model_conf['warmup_rate']
        self.grad_scaler = GradScaler(enabled=model_conf.get('mp_enabled', False))
        self.accumulation_steps = model_conf.get('accumulation_steps', 1)
        self.max_grad_norm = model_conf.get('max_grad_norm', None)
        self.swa_warmup = model_conf.get('swa_warmup', -1)
        self.swa_state = {}

        self.build_model()

        if os.path.exists(self.model_weight_path):
            self.load_model_parameters(self.model_weight_path)

        if self.beam_size > 1:
            self.test_batch_size = 1
            print("test_batch_size must be 1 if beam_size is bigger than 1")
        # assert self.test_batch_size == 1 or self.beam_size == 1, "test_batch_size must be 1 if beam_size is bigger than 1"

    def make_enc_dec_dict(self):
        ########## self.decode_only_input_token
        template_vocab_idx_in_bert = self.tokenizer.convert_tokens_to_ids(list(self.dataset.templatetoken2idx.keys())) # list of template token ids in BERT

        all_template_tokens = list(self.dataset.templatetoken2idx.keys())
        self.always_possible_tokens = ['[PAD]', '[BOS]', '[EOS]']
        for token in all_template_tokens:
            if token.startswith('[C'):
                self.always_possible_tokens.append(token)
            elif token.startswith('[OP'):
                self.always_possible_tokens.append(token)
        self.always_possible_tokens_idx = [self.dataset.templatetoken2idx[token] for token in self.always_possible_tokens]
        self.always_possible_tokens_idx = torch.tensor(self.always_possible_tokens_idx).to(self.device)

        self.dec_vocab_idx2enc_vocab_idx = {i: template_vocab_idx_in_bert[i] for i in self.dataset.idx2templatetoken.keys()} # dict 
        self.enc_vocab_idx2dec_vocab_idx = {v: k for k, v in self.dec_vocab_idx2enc_vocab_idx.items()} # dict 
        ########## self.decode_only_input_token

    def build_model(self):
        # BERT Encoder
        self.bert = AutoModel.from_pretrained(self.pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        print(self.tokenizer)
        # Add New Tokens (장소, 명칭, 인물)
        new_tokens = '(가) (나) (다) (라) (마) (바) (사) (아) (자) (차) (카) (타) (파) (하) 리터 l 밀리리터 ml 킬로미터 km 미터 m 센티미터 cm kg 제곱센티미터 ㎠ 세제곱센티미터 제곱미터 세제곱미터 ㎡ ㎤ ㎥'.split(' ')
        # Add New Tokens for Encoding Template using BERT
        new_tokens += list(self.dataset.templatetoken2idx.keys())
        num_added_toks = self.tokenizer.add_tokens(new_tokens)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        print(f'{num_added_toks} tokens are added!; {new_tokens}')
        
        self.make_enc_dec_dict()

        # Transformer Decoder for Generating Template
        self.embedding = nn.Embedding(self.num_template_vocab, self.bert_conf.hidden_size, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.bert_conf.hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.bert_conf.hidden_size, nhead=self.decoder_num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.decoder_num_layers)

        # Linear Layers for Generating NET and Classifying OP
        self.template_prediction_layer = nn.Linear(self.bert_conf.hidden_size, self.num_template_vocab) 

        # Linear Layers for Classifying Category
        self.classify_lastop_layer = nn.Linear(self.bert_conf.hidden_size, self.num_template_vocab)

        self.CE_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore_index [PAD]

        ############################## Optimizer parameter 그룹 설정 ######################################
        no_decay = ["bias", "LayerNorm.weight"]
        decoder_list = ['embedding', 'pos_encoder', 'decoder_layer', 'transformer_decoder']
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": self.reg,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": self.reg,
                "lr": self.decoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": 0.0,
                "lr": self.decoder_lr,
            },
        ]
        self.optimizer = AdamW(param_groups)
        ###########################################################################################

        self.to(self.device)

    # INPUT: IMQ // OUTPUT: Template
    def forward(self, batch_question_id, batch_question_attention, batch_template_id, batch_template_attention, batch_lastop_id):
        '''
        batch_question_id: (batch, seq_len)
        batch_question_attention: (batch, seq_len)
        batch_template_id: (batch, seq_len)
        batch_template_attention: (batch, seq_len)
        '''
        # Encoding IMQ using BERT
        batch_imq = self.bert(batch_question_id, attention_mask=batch_question_attention)[0]  # (batch, seq_len, hidden_size)
        batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, batch, hidden_size)

        # Predict category
        batch_cls = batch_imq[0]  # (batch, hidden_size)
        lastop_logits = self.classify_lastop_layer(batch_cls)  # (batch, num_template_vocab)

        if self.use_category:
            # remove BOS token and add lastop token
            batch_template_id = torch.cat([batch_lastop_id.unsqueeze(1), batch_template_id[:, 1:]], dim=1)  # (batch, seq_len)

        # template prediction
        batch_template_id = batch_template_id.permute(1, 0)  # (seq_len, batch)
        tgt_mask = self.generate_square_subsequent_mask(batch_template_id.size(0)).to(self.device)  # Padding mask for template
        tgt = self.embedding(batch_template_id)
        tgt = self.pos_encoder(tgt)

        # (batch, seq_len, hidden)
        decoder_output = self.transformer_decoder(
            tgt=tgt,         # (tgt_seq_len, batch, hidden_size) // Target NET to generate
            memory=batch_imq,         # (seq_len, batch, hidden_size) // Input IMQ
            tgt_mask=tgt_mask,        # (tgt_seq_len, tgt_seq_len) // to avoid looking at the future tokens (the ones on the right)
            tgt_key_padding_mask=~batch_template_attention.bool(),     # (batch, tgt_seq_len) // to avoid working on padding, Padding mask for NET (!!!!!!!! 1 for masking !!!!!!!!)
            memory_key_padding_mask=~batch_question_attention.bool()   # (batch, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ (!!!!!!!! 1 for masking !!!!!!!!)
        )
        decoder_output = decoder_output.permute(1, 0, 2) # (batch, seq_len, hidden_size)
        logits = self.template_prediction_layer(decoder_output)  # (batch, seq_len, |V|)

        return logits, lastop_logits

    def train_model_per_batch(self, batch_imq_id, batch_imq_attention, batch_template_id, batch_template_attention,
                            batch_lastop_id,
                            scheduler=None):

        # self.optimizer.zero_grad()

        # Numpy To Tensor
        batch_imq_id = torch.LongTensor(batch_imq_id).to(self.device)
        batch_imq_attention = torch.LongTensor(batch_imq_attention).to(self.device)
        batch_template_id = torch.LongTensor(batch_template_id).to(self.device)
        batch_template_attention = torch.LongTensor(batch_template_attention).to(self.device)
        batch_lastop_id = torch.LongTensor(batch_lastop_id).to(self.device)

        with torch.cuda.amp.autocast(self.grad_scaler.is_enabled()):
            # Model Forward
            # (batch, seq_len, |V|)
            template_logits, logit_lastop = self.forward(batch_imq_id, batch_imq_attention, batch_template_id, batch_template_attention, batch_lastop_id)

            # Calculate Loss
            # Loss for template prediction
            batch_template_id = batch_template_id[:, 1:]  # remove target [BOS]
            batch_template_attention = batch_template_attention[:, 1:]  # remove target [BOS]
            template_logits = template_logits[:, :-1, :]  # remove input [EOS]
            active_loss = batch_template_attention.reshape(-1) == 1
            active_batch_template_id = batch_template_id.reshape(-1)[active_loss]
            active_template_logits = template_logits.reshape(-1, template_logits.shape[-1])[active_loss]
            loss_template = self.CE_loss(active_template_logits, active_batch_template_id)

            loss = loss_template

            if self.use_category:
                # for lastop
                loss_lastop = self.CE_loss(logit_lastop, batch_lastop_id)
                loss += loss_lastop

        loss = loss / self.accumulation_steps
        loss = self.grad_scaler.scale(loss)

        # Backward
        loss.backward()

        if (self.global_step + 1) % self.accumulation_steps == 0:
            if self.max_grad_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

        # Step
        # self.optimizer.step()
        if scheduler:
            scheduler.step()

        return loss

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        # Set experimental configuration
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        save_step = exp_config['save_step']
        verbose = exp_config['verbose']
        self.log_dir = logger.log_dir
        self.global_step = 0
        start = time()

        # get data from dataset class
        full_imq_text = dataset.idx2IMQ
        full_template = dataset.idx2template

        # linear learning rate scheduler
        scheduler = None
        if self.lr_schedule:
            if len(dataset.train_ids) % self.batch_size == 0:
                steps_per_epoch = len(dataset.train_ids) // self.batch_size
            else:
                steps_per_epoch = len(dataset.train_ids) // self.batch_size + 1
            scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=int(steps_per_epoch*self.warmup_rate*num_epochs),
                                                        num_training_steps=steps_per_epoch * num_epochs)
            print(f">>> Linear scheduling at {self.warmup_rate} : warm up {self.warmup_rate*num_epochs} epochs over {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            epoch_train_start = time()
            epoch_loss = 0.0
            batch_loader = DataBatcher(np.arange(len(dataset.train_ids)), batch_size=self.batch_size, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            self.train()

            if epoch - 1 == self.swa_warmup:
                self.swa_init()

            for b, batch_idx in enumerate(tqdm(batch_loader, desc="train_model_per_batch", dynamic_ncols=True)):
                # get batch data
                batch_indices = dataset.train_ids[batch_idx]
                batch_imq_text = [full_imq_text[i] for i in batch_indices]
                batch_template_text = [full_template[i] for i in batch_indices]

                # Tokenize IMQ
                batch_imq_token = self.tokenizer(batch_imq_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_imq_id, batch_imq_attention = batch_imq_token['input_ids'], batch_imq_token['attention_mask']

                if self.do_mask_imq:
                    batch_imq_id_new = []
                    for idx, s in enumerate(batch_indices):
                        # exclude special tokens (INC, IXC, IEC, ISC, SEP, CLS)
                        tmp_list = []
                        tmp_list += self.dataset.idx2INC[s]
                        tmp_list += self.dataset.idx2IXC[s]
                        tmp_list += self.dataset.idx2IEC[s]
                        tmp_list += self.dataset.idx2ISC[s]
                        special_tokens = self.tokenizer(' '.join(tmp_list), add_special_tokens=False)['input_ids']
                        special_tokens.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
                        special_tokens.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token))
                        
                        batch_imq_token_idx = batch_imq_id[idx]
                        
                        # Candidate token indices to be deleted
                        candidate_indices = np.where(~np.in1d(batch_imq_token_idx, special_tokens) * batch_imq_attention[idx])[0]
                        
                        # Mask IMQ
                        mask_indices = np.random.choice(candidate_indices, size=int(len(candidate_indices)*self.imq_mask_ratio), replace=False)
                        batch_imq_token_idx[mask_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                        batch_imq_id_new.append(batch_imq_token_idx)
                        
                    batch_imq_id = np.array(batch_imq_id_new)
                    
                # Get template id, attention, lastop
                max_template_len = max([len(t.split()) for t in batch_template_text]) + 5
                batch_template_id = np.zeros((len(batch_template_text), max_template_len), dtype=np.int64)
                batch_template_attention = np.zeros_like(batch_template_id)
                batch_lastop_id = np.zeros_like(batch_template_id)[:, 0]
                for i, template in enumerate(batch_template_text):
                    batch_template_id[i][0] = dataset.templatetoken2idx['[BOS]']  # Add [BOS]
                    batch_template_attention[i][0] = 1
                    for j, v in enumerate(template.split(' ')):
                        batch_template_id[i][j+1] = dataset.templatetoken2idx[v]
                        batch_template_attention[i][j+1] = 1
                    if j > 0:
                        batch_lastop_id[i] = dataset.templatetoken2idx[v]
                    else:
                        batch_lastop_id[i] = dataset.templatetoken2idx['[BOS]']
                    batch_template_id[i][j+2] = dataset.templatetoken2idx['[EOS]']  # Add [EOS] to the end of the target sent
                    batch_template_attention[i][j+2] = 1

                batch_loss = self.train_model_per_batch(batch_imq_id, batch_imq_attention, batch_template_id, batch_template_attention,
                                                        batch_lastop_id,
                                                        scheduler)
                epoch_loss += batch_loss
                self.global_step += 1

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % (epoch_loss/num_batches), 'train time=%.2f' % epoch_train_time]

            # ======================== Valid
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                epoch_eval_start = time()

                self.swa_step()
                self.swap_swa_params()

                valid_score = evaluator.evaluate(self, dataset)
                valid_score['train_loss'] = epoch_loss.item()
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                # Test at the end of each epoch
                for testset in dataset.testsets:
                    valid_score.update(evaluator.evaluate(self, dataset, testset))
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(self.log_dir, 'best_model.p'))
                        torch.save({
                            "epoch": epoch,
                            'swa_state': self.swa_state,
                            'global_step': self.global_step,
                            'rng_states': (torch.get_rng_state(), np.random.get_state(), random.getstate()),
                            'optim': self.optimizer.state_dict(),
                            'scaler': self.grad_scaler.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler else None,
                        }, os.path.join(self.log_dir, 'state.p'))

                    if not os.path.exists(os.path.join(self.log_dir, 'tokenizer')):
                        self.tokenizer.save_pretrained(os.path.join(self.log_dir, 'tokenizer'))
                    if not os.path.exists(os.path.join(self.log_dir, 'dataset.pkl')):
                        with open(os.path.join(self.log_dir, 'dataset.pkl'), 'wb') as f:
                            pickle.dump((dataset.netvocab2netidx, dataset.netidx2netvocab, dataset.operator2idx, dataset.idx2operator, dataset.templatetoken2idx, dataset.idx2templatetoken), f, protocol=4)

                self.swap_swa_params()

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]
                
            if epoch % save_step == 0:
                # save best parameters
                torch.save(self.state_dict(), os.path.join(self.log_dir, f'best_model_{epoch}.p'))
                torch.save({
                    "epoch": epoch,
                    'swa_state': self.swa_state,
                    'global_step': self.global_step,
                    'rng_states': (torch.get_rng_state(), np.random.get_state(), random.getstate()),
                    'optim': self.optimizer.state_dict(),
                    'scaler': self.grad_scaler.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                }, os.path.join(self.log_dir, f'state_{epoch}.p'))

                if not os.path.exists(os.path.join(self.log_dir, 'tokenizer')):
                    self.tokenizer.save_pretrained(os.path.join(self.log_dir, 'tokenizer'))
                if not os.path.exists(os.path.join(self.log_dir, 'dataset.pkl')):
                    with open(os.path.join(self.log_dir, 'dataset.pkl'), 'wb') as f:
                        pickle.dump((dataset.netvocab2netidx, dataset.netidx2netvocab, dataset.operator2idx, dataset.idx2operator, dataset.templatetoken2idx, dataset.idx2templatetoken), f, protocol=4)

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time

    # Genrate template
    def predict(self, mode='valid', pf_converter=None):
        self.eval()
        with torch.no_grad():
            if mode == 'valid':
                input_ids = self.dataset.valid_ids
            elif 'test' in mode or mode == 'problemsheet':
                test_num = self.dataset.testsets.index(mode)
                input_ids = self.dataset.test_ids[test_num]
            # ------------------ SUBMISSION ------------------ #
            elif mode in ['submit', 'debug']:
                input_ids = self.dataset.test_ids
            # ------------------ SUBMISSION ------------------ #
            # get data from dataset class
            full_imq_text = self.dataset.idx2IMQ

            eval_answer = None
            eval_equation = []
            eval_loss = torch.zeros(len(input_ids))

            batch_size = self.test_batch_size
            batch_loader = DataBatcher(np.arange(len(input_ids)), batch_size=batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc=f'{mode}..', dynamic_ncols=True)):
                # get batch data
                batch_indices = input_ids[batch_idx]
                batch_imq_text = [full_imq_text[i] for i in batch_indices]

                # Tokenizer IMQ
                batch_imq_token = self.tokenizer(batch_imq_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_imq_id, batch_imq_attention = batch_imq_token['input_ids'], batch_imq_token['attention_mask']
                # Numpy to Tensor
                batch_imq_id = torch.LongTensor(batch_imq_id).to(self.device)
                batch_imq_attention = torch.LongTensor(batch_imq_attention).to(self.device)

                # Get greedy input = Generate template
                if self.beam_size == 1:
                    batch_pred_template_id = self.generate_template_greedy(batch_imq_id, batch_imq_attention)
                else:
                    batch_pred_template_id = self.generate_template_beam(batch_imq_id, batch_imq_attention)
                    batch_indices = np.array(batch_indices.tolist() * len(batch_pred_template_id))
                    batch_imq_text *= len(batch_pred_template_id)

                # Convert predicted template index to template vocab
                batch_pred_eq = [[] for _ in range(len(batch_indices))]
                for b, template_text in enumerate(batch_pred_template_id):
                    for v in template_text:
                        batch_pred_eq[b].append(self.dataset.idx2templatetoken[v])

                # Convert predicted token to number if token is not [OP_XX]
                for b, toks in enumerate(batch_pred_eq):
                    for i, tok in enumerate(toks):
                        if tok.startswith('[N') and self.dataset.idx2INC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2INC[batch_indices[b]][tok]
                        elif tok.startswith('[C'):
                            batch_pred_eq[b][i] = tok[2:-1]  # [C999] -> 999
                        elif tok.startswith('[X') and self.dataset.idx2IXC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2IXC[batch_indices[b]][tok]
                        elif tok.startswith('[E') and self.dataset.idx2IEC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2IEC[batch_indices[b]][tok]
                        elif tok.startswith('[S') and self.dataset.idx2ISC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2ISC[batch_indices[b]][tok]

                batch_pred_eq = [' '.join(eq) for eq in batch_pred_eq]

                # Check net integrity
                if self.beam_size > 1:
                    batch_pred_eq_checked = []     
                    for idx, pred_eq in enumerate(batch_pred_eq):
                        try:
                            pf_converter.convert(pred_eq, batch_imq_text[idx])
                            batch_pred_eq_checked.append(pred_eq)
                            break
                        except:
                            continue                 
                    if batch_pred_eq_checked:
                        batch_pred_eq = [batch_pred_eq_checked[0]]
                    else:
                        batch_pred_eq = [batch_pred_eq[0]]

                # Concatenate
                eval_equation += batch_pred_eq
                # print("True:" ,[self.dataset.idx2postfix[input_ids[idx]] for idx in batch_idx])
                # print("Pred:" ,batch_pred_eq)
        return eval_answer, eval_equation, eval_loss.numpy()

    # INPUT: IMQ // OUTPUT: Template
    def generate_template_greedy(self, batch_question_id, batch_question_attention):
        '''
        batch_question_id (batch, max_seq_len)
        batch_question_attention (batch, max_seq_len)
        '''
        bos_idx = self.dataset.templatetoken2idx['[BOS]']
        eos_idx = self.dataset.templatetoken2idx['[EOS]']

        batch_size, max_seq_len = batch_question_id.size()
        pred_template = [[bos_idx] for _ in range(batch_size)]

        # Get BERT hidden state
        batch_imq = self.bert(batch_question_id, attention_mask=batch_question_attention)[0]  # (batch_size, seq_len, hidden_size)
        batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        
        if self.use_category:
            batch_cls = batch_imq[0]  # (batch, hidden_size)
            lastop_logits = self.classify_lastop_layer(batch_cls)  # (batch, num_template_token)
            lastop_pred = torch.argmax(lastop_logits, dim=-1)  # (batch)
            # remove BOS token and add lastop token
            pred_template = [[lastop_pred[b].item()] for b in range(batch_size)]

        cur_possible_tokens_idx = torch.zeros_like(batch_question_id)
        for i in range(batch_question_id.size(0)):
            for j in range(batch_question_id.size(1)):
                cur_possible_tokens_idx[i, j] = self.enc_vocab_idx2dec_vocab_idx.get(batch_question_id[i, j].item(), 0)

        for b in range(batch_size):
            for i in range(max_seq_len):
                tgt = torch.LongTensor([pred_template[b]]).view(-1, 1).to(self.device)  # (cur_seq_len, 1)
                tgt_mask = self.generate_square_subsequent_mask(i+1).to(self.device)  # (cur_seq_len, cur_seq_len)
                tgt = self.embedding(tgt)  # (cur_seq_len, 1, hidden)
                tgt = self.pos_encoder(tgt)  # (cur_seq_len, 1, hidden)

                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=batch_imq[:, b, :].unsqueeze(1),
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~batch_question_attention[b].unsqueeze(0).bool()   # (1, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ
                ).permute(1, 0, 2)

                # decoder_output: (1, cur_seq_len, hidden)
                tmp_logit_template = self.template_prediction_layer(decoder_output)
                logits = tmp_logit_template[:, -1, :]  # the last timestep # (1, |V|)

                if self.decode_only_input_token:
                    # remove tokens which not in input imq##############################################################
                    new_logits = torch.zeros_like(logits) - 1e10 # (1, |V|)
                    new_logits[0, self.always_possible_tokens_idx] = logits[0, self.always_possible_tokens_idx] # (1, |V|)
                    new_logits[0, cur_possible_tokens_idx[b]] = logits[0, cur_possible_tokens_idx[b]]
                    logits = new_logits
                    # remove tokens which not in input imq##############################################################

                indices = logits.argmax(dim=-1)  # (1, 1)
                pred_template[b].append(indices.item())

                if indices.item() == eos_idx:  # break if end token appears
                    break
            pred_template[b] = pred_template[b][1:-1]  # append without [BOS], [EOS] token

        return pred_template

    def _get_length_penalty(self, length, alpha=0.5, min_length=5):
        return ((min_length + length) / (min_length+1)) ** alpha


    def generate_template_beam(self, batch_question_id, batch_question_attention, isEmbed=False):
        '''
        batch_question_id (batch, max_seq_len)
        batch_question_attention (batch, max_seq_len)
        '''
        bos_idx = self.dataset.templatetoken2idx['[BOS]']
        eos_idx = self.dataset.templatetoken2idx['[EOS]']

        pred_template = [[bos_idx] * self.beam_size for _ in range(batch_question_id.size(0))]
        batch_size, _ = batch_question_id.size()
        max_seq_len = 50
        for b in range(batch_size):
            # Get BERT hidden state
            self.eval()
            beam_size = self.beam_size
            batch_imq = self.bert(batch_question_id[b].unsqueeze(0), attention_mask=batch_question_attention[b].unsqueeze(0))[0]  # (1, seq_len, hidden_size)
            batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, 1, hidden_size)
            cumulative_probs = torch.Tensor([0] * beam_size)
            pred_template_beam = np.array([1]).repeat(beam_size*beam_size).reshape(-1, 1)
            candidate_template = []

            if self.use_category:
                batch_cls = batch_imq[0]  # (1, hidden_size)
                lastop_logits = self.classify_lastop_layer(batch_cls)  # (1, num_op_vocab)
                lastop_pred = torch.argmax(lastop_logits, dim=-1)  # (1)
                # remove BOS token and add lastop token
                pred_template = [[lastop_pred[b].item()] for b in range(batch_size)]

            cur_possible_tokens_idx = torch.zeros_like(batch_question_id[b]) # (seq_len)
            for j in range(batch_question_id.size(1)):
                cur_possible_tokens_idx[j] = self.enc_vocab_idx2dec_vocab_idx.get(batch_question_id[b, j].item(), 0)

            for i in range(max_seq_len):
                tgt = torch.LongTensor([pred_template[b]]).view(-1, beam_size).to(self.device)  # (cur_seq_len, beam)
                tgt_mask = self.generate_square_subsequent_mask(i+1).to(self.device)  # (cur_seq_len, cur_seq_len)
                tgt = self.embedding(tgt)  # (cur_seq_len, hidden) -> (cur_seq_len, beam, hidden_size)
                tgt = self.pos_encoder(tgt)  # (cur_seq_len, 1, hidden)
                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=batch_imq.repeat(1, beam_size, 1),
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~batch_question_attention[b].unsqueeze(0).repeat(beam_size, 1).bool()   # (1, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ
                ) # (cur_seq_len, beam, hidden_size)

                tmp_logit_template = self.template_prediction_layer(decoder_output)  # (cur_seq_len, beam, hidden_size) -> (cur_seq_len, beam, |V|)
                logits = tmp_logit_template[-1, :, :]  # the last timestep # (beam, |V|)

                if self.decode_only_input_token:
                    # remove tokens which not in input imq##############################################################
                    new_logits = torch.zeros_like(logits) - 1e10 # (beam, |V|)
                    new_logits[:, self.always_possible_tokens_idx] = logits[:, self.always_possible_tokens_idx] # (beam, |V|)
                    new_logits[:, cur_possible_tokens_idx] = logits[:, cur_possible_tokens_idx] # (beam, |V|)
                    logits = new_logits
                    # remove tokens which not in input imq##############################################################

                if i == 0:
                    logits[:, 2] = -np.inf
                values, indices = F.softmax(logits, dim=-1).topk(beam_size, dim=-1, sorted=False)
                indices = indices.cpu().reshape(-1, 1)
                if i == max_seq_len-1:
                    indices[:] = 2  # 2 * beam_size * beam_size
                    values[:] = 1

                pred_template_beam = np.hstack([pred_template_beam, indices])
                cumulative_probs = (cumulative_probs.view(-1, 1).repeat(1, beam_size) - (-np.log(values.cpu()))).flatten()

                if i == 0: ### [BOS] token
                    _, beam_search_idx = cumulative_probs[:beam_size].topk(beam_size)
                else:
                    _, beam_search_idx = cumulative_probs.topk(beam_size)

                if len(torch.where(indices == eos_idx)[0]) > 0:
                    top = 0
                    while top < beam_size:
                        top_idx = beam_search_idx[top]
                        if indices[top_idx].item() == eos_idx:
                            candidate_idx = beam_search_idx[top]
                            template = pred_template_beam[candidate_idx]
                            final_val = cumulative_probs[candidate_idx] / self._get_length_penalty(template.shape[0])
                            ## append top1
                            candidate_template.append((template, final_val))
                            top += 1
                        else:
                            break
                    temp_indices, _ = torch.where(indices == eos_idx)
                    cumulative_probs[temp_indices] = -np.inf
                    _, beam_search_idx = cumulative_probs.topk(beam_size)

                ## 끝나는 지점
                if len(candidate_template) > beam_size * 3 or i == max_seq_len-1:
                    if isEmbed:
                        embed()
                    pred_template_beam = [template_score[0].tolist()[1:-1] for template_score in sorted(candidate_template, key=(lambda x:x[1]), reverse=True)]
                    if pred_template_beam:
                        pred_template = pred_template_beam
                    else:
                        pred_template = []
                    return pred_template

                pred_template[b] = pred_template_beam[beam_search_idx].reshape(beam_size, -1).T ## (cur_seq_len + 1, beam_size)
                pred_template_beam = pred_template[b].repeat(beam_size, axis=-1).T
                cumulative_probs = cumulative_probs[beam_search_idx]

        return []

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def load_model_parameters(self, log_dir):
        # model
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        try:
            self.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {log_dir}.")
        except RuntimeError:
            del state_dict['bert.embeddings.word_embeddings.weight']
            del state_dict['embedding.weight']
            del state_dict['template_prediction_layer.weight']
            del state_dict['template_prediction_layer.bias']
            self.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {log_dir}. But bert.embeddings.word_embeddings, embedding, and template_prediction_layer are not loaded!")

    def restore(self, log_dir):
        self.log_dir = log_dir
        # model
        # ------------------ SUBMISSION ------------------ #
        with open(os.path.join(self.log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        self.to(self.device)
        # ------------------ SUBMISSION ------------------ #
        # Get template vocab size and op vocab size and rebuild model
        self.num_template_vocab = state_dict['template_prediction_layer.weight'].shape[0]
        self.build_model()
        # tokenizer
        self.tokenizer = self.tokenizer.from_pretrained(os.path.join(self.log_dir, 'tokenizer'))
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.load_state_dict(state_dict)
        # Dataset
        with open(os.path.join(self.log_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset.netvocab2netidx, self.dataset.netidx2netvocab, self.dataset.operator2idx, self.dataset.idx2operator, self.dataset.templatetoken2idx, self.dataset.idx2templatetoken = pickle.load(f)
        self.make_enc_dec_dict()

    def swa_init(self) -> None:
        self.swa_state["models_num"] = 1
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.clone().detach()

    def swa_step(self) -> None:
        if not self.swa_state:
            return

        self.swa_state["models_num"] += 1
        beta = 1.0 / self.swa_state["models_num"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)

    def swap_swa_params(self) -> None:
        if not self.swa_state:
            return

        for n, p in self.named_parameters():
            p.data, self.swa_state[n] = self.swa_state[n], p.data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
