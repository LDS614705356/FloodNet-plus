import torch
from torch import nn
# from engine.utils import NestedTensor
from models.caption.base import BaseCaptioner
from einops import rearrange, repeat


class Transformer(BaseCaptioner):

    def __init__(self,
                 junction,
                 grid_fw,
                 cap_generator,
                 bos_idx=2,
                 use_gri_feat=True,
                 use_reg_feat=False,
                 cached_features=False):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        # self.grid_net = grid_net
        self.junction = junction
        self.grid_fw = grid_fw
        self.cap_generator = cap_generator
        self.use_reg_feat = use_reg_feat
        self.use_gri_feat = use_gri_feat
        self.cached_features = cached_features
        # self.config = config

        if self.use_gri_feat:
            self.register_state('gri_feat', None)
            # self.register_state('gri_mask', None)

        # if self.use_reg_feat:
            # self.register_state('reg_feat', None)
            # self.register_state('reg_mask', None)

        self.init_weights()
        # self.detector = detector

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_bs_device(self, samples):

            # key = 'gri_feat' if 'gri_feat' in samples else 'reg_feat'
        batch_size = samples.shape[0]
        device = samples.device
        # elif isinstance(samples, NestedTensor):
        #     batch_size = samples.tensors.shape[0]
        #     device = samples.tensors.device
        return batch_size, device

    def init_state(self, batch_size, device):
        return [torch.zeros((batch_size, 0), dtype=torch.long, device=device), None, None]

    def select(self, t, candidate_logprob, beam_size, **kwargs):
        candidate_logprob = rearrange(candidate_logprob, 'B Beam V -> B (Beam V)')
        selected_logprob, selected_idx = torch.sort(candidate_logprob, -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size] # select top beam_size ones
        return selected_idx, selected_logprob  # [B Beam]

    def _expand_state(self, selected_beam, cur_beam_size, batch_size, beam_size):

        def fn(tensor):
            shape = [int(sh) for sh in tensor.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            tensor = torch.gather(tensor.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                                  beam.expand(*([batch_size, beam_size] + shape[1:])))
            tensor = tensor.view(*([-1] + shape[1:]))
            return tensor

        return fn # return a special function

    def forward(self,
                images,
                seq,
                seq_for_softmax_loss=None,
                use_beam_search=False,
                max_len=20,
                eos_idx=3,
                beam_size=5,
                out_size=1,
                question=None,
                return_probs=False,
                **kwargs):
        if not use_beam_search:
            # if not self.cached_features:
            #     vis_inputs = self.detector(images)
            # else:
            #     vis_inputs = images

            images = self.junction(images)
            images, _ = self.grid_fw(images, question)
            # if self.use_gri_feat:
            #     gri_feat, _ = self.grid_net(vis_inputs['gri_feat'], attention_mask=vis_inputs['gri_mask'])
            vis_inputs = images

            dec_output = self.cap_generator(seq, vis_inputs)
            return dec_output
        else:  # Run beam_search in the following code
            batch_size, device = self.get_bs_device(images)

            # the mask of the current word (whether it != eos or not), it = 1 if != <eos>
            self.seq_mask = torch.ones((batch_size, beam_size, 1), device=device)

            # the cummulative sum of log probs up to the current word [B, Beam, 1]
            self.seq_logprob = torch.zeros((batch_size, 1, 1), device=device)

            # log probs of all beam_size selected words: [[B, Beam, 1] * max_len]
            self.log_probs = []
            self.selected_words = None

            if return_probs:
                self.all_log_probs = []

            self.gt_softmax_loss = []

            # selected words at each timestep: [[B, Beam, 1] * max_len]
            outputs = []

            with self.statefulness(batch_size):
                for timestep in range(max_len):
                    images, outputs = self.iter(
                        timestep=timestep,
                        samples=images,
                        outputs=outputs,
                        question=question,
                        return_probs=return_probs,
                        batch_size=batch_size,
                        beam_size=beam_size,
                        eos_idx=eos_idx,
                        seq_for_softmax_loss=seq_for_softmax_loss,
                        **kwargs,
                    )

            # Sort result
            seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)

            # sum log_probs = seq_logprob
            # outputs = log_probs shape = [B, Beam, Len], the following is to sorted the order of which sequence.
            outputs = torch.cat(outputs, -1)  # [B, Beam, Len]
            outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            log_probs = torch.cat(self.log_probs, -1)
            log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            if seq_for_softmax_loss != None:
                self.gt_softmax_loss = [repeat(o, "B Beam eg -> B Beam eg V", V=1) for o in self.gt_softmax_loss]
                softmax_loss = torch.cat(self.gt_softmax_loss, -1)
                softmax_loss = torch.sum(softmax_loss, dim=-1)/max_len
                softmax_loss = torch.gather(softmax_loss, 1, sort_idxs.expand(batch_size, beam_size, seq_for_softmax_loss.shape[1]))
            if return_probs:
                all_log_probs = torch.cat(self.all_log_probs, 2)
                all_log_probs = torch.gather(
                    all_log_probs, 1,
                    sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, max_len, all_log_probs.shape[-1]))

            outputs = outputs.contiguous()[:, :
                                           out_size]  # [B Beam Len] -> [B, :topk, Len] select only the top k sentences
            log_probs = log_probs.contiguous()[:, :out_size]  # [B Beam Len] -> [B Len] select only the top k sentences
            if seq_for_softmax_loss != None:
                weights = torch.tensor([1, 0.8, 0.6, 0.4, 0.2])
                weights = repeat(weights, 'Beam -> B Beam eg', B=softmax_loss.shape[0], eg=softmax_loss.shape[2])
                weights = weights.to(softmax_loss.device)
                # weights = repeat(weights, 'B Beam eg -> B eg Beam')
                # weights = torch.reshape(weights, )
                softmax_loss = softmax_loss.contiguous()[:, :out_size]
                # order = torch.max(softmax_loss, 1)[1]

                fd = torch.tensor([i for i in range(softmax_loss.shape[0])])
                fd = repeat(fd, "dim1 -> dim1 dim2", dim2=softmax_loss.shape[2])
                # f = rearrange(f, "dim1 dim2 -> dim2 dim1")
                fd = rearrange(fd, "dim1 dim2 -> (dim1 dim2)")
                order = torch.max(softmax_loss, 1)[1]
                order = rearrange(order, "dim1 dim2 -> (dim1 dim2)")
                td = torch.tensor([i for i in range(softmax_loss.shape[2])])
                td = repeat(td, "dim2 -> dim1 dim2", dim1=softmax_loss.shape[0])
                td = rearrange(td, "dim1 dim2 -> (dim1 dim2)")
                # order = torch.unsqueeze(order, 2)

                softmax_loss[fd, order, td] = -softmax_loss[fd, order, td]

                # softmax_loss[order] = -softmax_loss[order]
                # order = order * 0.2 + 0.2  # weight
                softmax_loss = softmax_loss * weights

            if out_size == 1:
                outputs = outputs.squeeze(1)  # [B :topk, len] = [B, len] if topk = 1
                log_probs = log_probs.squeeze(1)

            if return_probs:
                return outputs, log_probs, all_log_probs
            elif seq_for_softmax_loss != None:
                return outputs, log_probs, softmax_loss
            else:
                return outputs, log_probs, None

    def step(self, timestep, prev_output, samples, seq, mode='teacher_forcing', question=None, **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if timestep == 0:
                samples = self.junction(samples)
                samples, _ = self.grid_fw(samples, question)
                self.gri_feat = samples
                # if not self.cached_features:
                #     vis_inputs = self.detector(samples)
                # else:
                #     vis_inputs = samples
                #
                # if self.config.model.use_gri_feat:
                #     self.gri_feat, self.gri_mask = self.grid_net(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                #     self.gri_feat = self.gri_feat[:, -1]
                #
                # if self.config.model.use_reg_feat:
                #     self.reg_feat = vis_inputs['reg_feat']
                #     self.reg_mask = vis_inputs['reg_mask']

                # If t = 0, enc_output = [B, N, D], init_tokens = [B, 1]
                # Else t > 0, enc_output = [BB, N, D], it = prev_output (t-1) = [BB, 1]
                # _feat = getattr(self, 'gri_feat', self.reg_feat)
                _feat = getattr(self, 'gri_feat', self.gri_feat)
                it = _feat.data.new_full((_feat.shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        vis_inputs = None
        if self.use_gri_feat:
            # vis_inputs['gri_feat'] = self.gri_feat
            # vis_inputs['gri_mask'] = self.gri_mask
            # vis_inputs['gri_mask'] = None
            vis_inputs = self.gri_feat

        # if self.config.model.use_reg_feat:
        #     vis_inputs['reg_feat'] = self.reg_feat
        #     vis_inputs['reg_mask'] = self.reg_mask

        return self.cap_generator(it, vis_inputs)

    def iter(self, timestep, samples, question, outputs, return_probs, batch_size, seq_for_softmax_loss, beam_size=5, eos_idx=3, **kwargs):
        cur_beam_size = 1 if timestep == 0 else beam_size

        word_logprob = self.step(timestep, self.selected_words, samples, None, mode='feedback', question=question, **kwargs)
        if seq_for_softmax_loss != None:
            gt = seq_for_softmax_loss[:,:,timestep]
            gt = repeat(gt, 'B eg -> B eg Beam', Beam=beam_size)
            gt = rearrange(gt, 'B eg Beam -> B Beam eg')
        # gt_mask = (gt != 1)

        # softmax_logprob =
        word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)  # [BB V] -> [B Beam V] V!=1
        candidate_logprob = self.seq_logprob + word_logprob  # [B Beam V] # add to get the total score
        word_logprob_for_gt = word_logprob.contiguous()


        if timestep == 0 and seq_for_softmax_loss != None:
            gt_softmax = repeat(word_logprob_for_gt[:,:,2], 'B C->B (n C) V', n=beam_size, V=gt.size(1)) # B, each Beam, each gt
            self.gt_softmax_loss.append(gt_softmax)

        # Mask sequence if it reaches EOS
        if timestep > 0:
            _selected_words = self.selected_words.view(batch_size, cur_beam_size)  # [BB, 1] -> [B Beam]
            # mask = 0 if it is eos, else 1.
            mask = repeat((_selected_words != eos_idx).float(), 'B Beam -> B Beam V', V=1)
            self.seq_mask = self.seq_mask * mask  # [B Beam V] V=1 #if used to be <eos>, all words after are <unk>
            word_logprob = word_logprob * self.seq_mask  # [B Beam V] V!=1
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 2:] = -999  # [B Beam V] ##### [B Beam V] V!=1 <pad>!=-999
            old_seq_logprob[:, :, 0] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)  # [B Beam V] #if <eos>, the score doesn't add
            # After <EOS>, we want to make all predictions to <PAD>.
            # When decoding, we will remove all predictions after <EOS>

        selected_idx, selected_logprob = self.select(timestep, candidate_logprob, beam_size, **kwargs) # select top beam_size ones
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1],  rounding_mode='floor')  # [B Beam]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]  # [B Beam] # change selected_words to the range[0, V],V!=1

        # save the states of the selected beam
        self.apply_to_states(self._expand_state(selected_beam, cur_beam_size, batch_size, beam_size))  # [BB, ...]
        # beam:第几个分支，log_prob:分数，
        self.seq_logprob = repeat(selected_logprob, 'B Beam -> B Beam L', L=1)
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.seq_mask = torch.gather(self.seq_mask, 1, beam_exp)
        outputs = [torch.gather(o, 1, beam_exp) for o in outputs]
        outputs.append(repeat(selected_words, 'B Beam -> B Beam L', L=1))
        if timestep > 0 and seq_for_softmax_loss != None:
            beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob_for_gt.shape[-1])
            word_beam_prob = torch.gather(word_logprob_for_gt, 1, beam_exp)
            gt_softmax = torch.gather(word_beam_prob, 2, gt) #single word's softmax loss
            beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=gt.shape[-1])
            # gt_softmax = gt_softmax * gt_mask
            # gt_softmax_loss = self.gt_softmax_loss
            self.gt_softmax_loss = [torch.gather(o, 1, beam_exp) for o in self.gt_softmax_loss]
            self.gt_softmax_loss.append(gt_softmax)


        if return_probs:
            if timestep == 0:
                # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.expand((batch_size, beam_size, -1)).unsqueeze(2))
            else:  # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob.shape[-1])
        this_word_logprob = torch.gather(word_logprob, 1, beam_exp)
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.log_probs = [torch.gather(o, 1, beam_exp) for o in self.log_probs] #choose the beam

        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)  # [B*Beam, 1]

        return samples, outputs
