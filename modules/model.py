import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, LogitsProcessorList
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform, BertOnlyMLMHead
from modules.config import MVLBertConfig
from transformers import BeamSearchScorer

import random
from modules.visual_feature_extractor import resnet101_without_fc, resnet50_without_poolfc, linear_patch_16x16, \
    SwinTransformer, VisionTransformerBaseWithoutPooling

from modules.swin_transformer_config import parse_option


class MVLBert(nn.Module):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__()
        self.config_class = MVLBertConfig
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = BertEncoder(config)
        self.is_decoder = self.config.is_decoder
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # self.register_buffer("position_ids", torch.arange(7*7*2 + 3 + config.max_length).expand((1, -1)))
        self.register_buffer("position_ids", torch.arange(512).expand((1, -1)))

    def forward(self, text_idx, text_mask, image_feature, image_mask, past_key_values=None, use_cache=False,
                seq2seq_mask=False, output_text_image_seperate=False):

        embedding_output, attention_mask, obj_end, text_end = \
            self.get_embedding(text_idx, text_mask, image_feature, image_mask, seq2seq_mask=seq2seq_mask,
                               past_key_values=past_key_values)
        # print("attention_mask:", attention_mask)
        # print("attention_mask.shape:", attention_mask.shape)
        # print("text_mask_new:", text_mask_new)
        # print("text_mask_new.shape:", text_mask_new.shape)
        # print("object_mask_new:", object_mask_new)
        # print("object_mask_new.shape:", object_mask_new.shape)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask) if attention_mask is not None else None
        # print("embedding_output.shape:", embedding_output.shape)
        # print("extended_attention_mask.shape:", extended_attention_mask.shape)

        # BaseModelOutputWithPastAndCrossAttentions(
        #             last_hidden_state=hidden_states,
        #             past_key_values=next_decoder_cache,
        #             hidden_states=all_hidden_states,
        #             attentions=all_self_attentions,
        #             cross_attentions=all_cross_attentions,
        #         )
        encoder_output = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache)
        last_layer_output = encoder_output[0]  # last_hidden_state
        pooler_output = self.pooler(last_layer_output) if self.pooler is not None else None
        # last_layer_output: (batch, seq_len, hidden_size)
        if output_text_image_seperate:
            image_output = last_layer_output[:, 1:obj_end]
            # print("encoder_text_output:", encoder_text_output.shape)
            text_output = last_layer_output[:, obj_end + 1: text_end]
            sep_output = last_layer_output[:, obj_end]
            return text_output, image_output, pooler_output, sep_output

        return encoder_output, pooler_output

    def get_embedding(self, text_idx, text_mask, image_feature, image_mask, seq2seq_mask=False, past_key_values=None):
        # combine get word embedding, position embedding, type_embedding, image_feature
        # text_idx: (batch, seq_len)   t1, t2, ..., tn, ([END], [PAD], [PAD], ... if it's an encoder)
        # text_mask: 0 for [PAD], 1 for other
        # image_feature: (batch, h*w, channel)
        # Note: text_idx and text_mask can be None in generation (report generation & image captioning) task
        #       in the first run of the generation loop
        batch = image_feature.shape[0]
        if past_key_values is not None:
            # batch, num_attention_heads, image_text_seq_len, attention_hidden_dim
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length = text_idx.shape[1]
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

            text_embedding = self.word_embeddings(text_idx)
            position_embeddings = self.position_embeddings(position_ids)

            token_type_ids = torch.zeros_like(text_idx)
            token_type_embedding = self.token_type_embeddings(token_type_ids)
            embedding_output = text_embedding + token_type_embedding + position_embeddings
            # print("past_key_values[0][0].shape", past_key_values[0][0].shape)
            # print("text_idx.shape", text_idx.shape)
            if seq2seq_mask:
                max_length = past_key_values_length + seq_length
                row_inc, col_inc = torch.meshgrid(
                    [torch.arange(max_length, dtype=torch.int64, device=image_feature.device),
                     torch.arange(max_length, dtype=torch.int64, device=image_feature.device)])
                seq2seq_attention_mask = col_inc <= row_inc
                seq2seq_attention_mask_new = seq2seq_attention_mask[-2:]
                # seq2seq_attention_mask_new = seq2seq_attention_mask[-text_idx.shape[1]:]  # if text_idx.shape[1] > 1 else seq2seq_attention_mask[-1:]
                attention_mask = seq2seq_attention_mask_new[None, :, :].repeat(batch, 1, 1)
            else:
                attention_mask = None

            return embedding_output, attention_mask, position_ids[:, -1], position_ids[:, -1] + 1

        cls_id, sep_id = self.config.cls_token_id, self.config.sep_token_id  # self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=image_feature.device)
        sep_mask = torch.ones((batch, 1), dtype=torch.bool, device=image_feature.device)
        obj_end = image_feature.shape[1] + 1  # index of [SEP]
        seq_len = text_mask.shape[1] if text_mask is not None else 0
        text_end = obj_end + seq_len + 1  # next idx of last token of caption
        max_length = seq_len + image_mask.shape[1] + 2  # (batch, num_obj + seq_len + 2)

        if seq2seq_mask:
            row_inc, col_inc = torch.meshgrid([torch.arange(max_length, dtype=torch.int64, device=image_feature.device),
                                               torch.arange(max_length, dtype=torch.int64, device=image_feature.device)])
            seq2seq_attention_mask = col_inc <= row_inc
            seq2seq_attention_mask[col_inc <= obj_end] = 1
            attention_mask = seq2seq_attention_mask[None, :, :].repeat(batch, 1, 1)
        else:
            if text_mask is not None:
                attention_mask = torch.cat([cls_mask, image_mask, sep_mask, text_mask], dim=1)
            else:
                attention_mask = torch.cat([cls_mask, image_mask, sep_mask], dim=1)

        # torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)
        # torch.set_printoptions(edgeitems=768)
        # print(attention_mask.shape)
        cls_tensor = torch.ones((batch, 1), dtype=torch.int64, device=image_feature.device).fill_(cls_id)
        cls_embedding = self.word_embeddings(cls_tensor)
        sep_tensor = torch.ones((batch, 1), dtype=torch.int64, device=image_feature.device).fill_(sep_id)
        sep_embedding = self.word_embeddings(sep_tensor)
        if text_idx is not None:
            text_embedding = self.word_embeddings(text_idx)
            # (batch, obj_num+seq_len+2, hidden)
            # if text_idx.shape[0][1] != sep_id:
            vl_embedding = torch.cat([cls_embedding, image_feature, sep_embedding, text_embedding], dim=1)
            # else:
            #     # for report generation in `normal` learning strategy with beam search.
            #     vl_embedding = torch.cat([cls_embedding, image_feature, text_embedding], dim=1)
        else:
            # (batch, obj_num+2, hidden)
            vl_embedding = torch.cat([cls_embedding, image_feature, sep_embedding], dim=1)

        grid_ind, grid_pos = torch.meshgrid([torch.arange(batch, dtype=torch.long, device=image_feature.device),
                                             torch.arange(max_length, dtype=torch.long, device=image_feature.device)])

        token_type_ids = torch.zeros((batch, max_length), dtype=torch.int64, device=image_feature.device)
        token_type_ids[grid_pos <= obj_end] = 1  # 0 for text, 1 for image
        token_type_embedding = self.token_type_embeddings(token_type_ids)

        position_ids = grid_pos
        position_embeddings = self.position_embeddings(position_ids)
        embedding_output = vl_embedding + token_type_embedding + position_embeddings

        return embedding_output, attention_mask, obj_end, text_end

    def get_extended_attention_mask(self, attention_mask):
        # provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # print(f"extended_attention_mask, {attention_mask.shape}:")
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            assert False, "error attention mask shape!"

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # TODO: add fp16?
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # print(next(self.parameters()).dtype)
        # print(f"extended_attention_mask, {extended_attention_mask.shape}:")
        # print(f"{extended_attention_mask}")
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class Conv_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.vgg16 = myvgg(True, True)
        # a: torch.Size([7,7,512])
        # self.resnet101 = resnet101_without_fc(True, True)
        self.config = config
        self.hidden_size = config.hidden_size if config is not None else 768
        print("conv backbone:", config.conv)
        if config.conv == 'resnet101':
            PRETRAIN = True
            print("Resnet101 PRETRAIN:", PRETRAIN)
            # conv = resnet101_without_fc(True, True)
            conv = resnet101_without_fc(PRETRAIN, PRETRAIN)
        elif config.conv == 'linear':
            conv = linear_patch_16x16()
        elif config.conv == 'resnet50':
            conv = resnet50_without_poolfc()
        elif config.conv.lower() == 'swintransformer':
            _, swin_config = parse_option()
            conv = SwinTransformer(img_size=swin_config.DATA.IMG_SIZE,
                                   patch_size=swin_config.MODEL.SWIN.PATCH_SIZE,
                                   in_chans=swin_config.MODEL.SWIN.IN_CHANS,
                                   num_classes=swin_config.MODEL.NUM_CLASSES,
                                   embed_dim=swin_config.MODEL.SWIN.EMBED_DIM,
                                   depths=swin_config.MODEL.SWIN.DEPTHS,
                                   num_heads=swin_config.MODEL.SWIN.NUM_HEADS,
                                   window_size=swin_config.MODEL.SWIN.WINDOW_SIZE,
                                   mlp_ratio=swin_config.MODEL.SWIN.MLP_RATIO,
                                   qkv_bias=swin_config.MODEL.SWIN.QKV_BIAS,
                                   qk_scale=swin_config.MODEL.SWIN.QK_SCALE,
                                   drop_rate=swin_config.MODEL.DROP_RATE,
                                   drop_path_rate=swin_config.MODEL.DROP_PATH_RATE,
                                   ape=swin_config.MODEL.SWIN.APE,
                                   patch_norm=swin_config.MODEL.SWIN.PATCH_NORM,
                                   use_checkpoint=swin_config.TRAIN.USE_CHECKPOINT)
            resume_path = './modules/swin_small_patch4_window7_224.pth'
            # resume_path = './checkpoints/swin_base_patch4_window7_224.pth'
            checkpoint = torch.load(resume_path, map_location='cpu')
            conv.load_state_dict(checkpoint['model'], strict=False)
            print("load swin-transformer weight from", resume_path)
        elif config.conv.lower() == 'vit' or config.conv.lower() == 'visiontransformer':
            conv = VisionTransformerBaseWithoutPooling(True, True)
        else:
            raise NotImplementedError('no such config.conv')

        self.conv = nn.Sequential(
            conv,
            nn.GELU(),
        )
        self.resnet_fc = nn.Linear(2048, config.hidden_size)

    def forward(self, v):
        # print("input v shape:", v.shape)
        if torch.is_tensor(v) and len(v.shape) == 5:
            # for iu-xray
            # v: batch, 2, channel, h, w
            v1 = v[:, 0]
            v2 = v[:, 1]
            obj1 = self.conv(v1)
            obj2 = self.conv(v2)
            shape = obj1.shape

            if len(shape) == 4:
                obj1 = torch.reshape(obj1, (shape[0], shape[1], shape[2] * shape[3])).transpose(1, 2)
                obj2 = torch.reshape(obj2, (shape[0], shape[1], shape[2] * shape[3])).transpose(1, 2)
            # batch, obj_num, channel
            obj = torch.cat((obj1, obj2), dim=1)

        else:
            obj = self.conv(v)
            shape = obj.shape
            # print("shape:", shape)
            if len(shape) == 4:
                # batch, channel, h, w -> batch, obj_num, channel
                obj = torch.reshape(obj, (shape[0], shape[1], shape[2] * shape[3])).transpose(1, 2)
        # batch, c, h, w  ->  batch, channel, obj_num
        if obj.shape[2] == 2048:
            obj = self.resnet_fc(obj)

        return obj


class MVLBertPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    Modified from BertPreTrainedModel from huggingface
    """

    base_model_prefix = "MVLBert"
    config_class = MVLBertConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MVLBertForVQA(MVLBertPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # from RelationNetwork: image_size 128 x 128,
        # 4 convolutional layers each with 24 kernels, ReLU non-linearities, and batch normalization
        # a three-layer MLP consisting of 256, 256 (with 50% dropout), and
        # 29 units with ReLU non-linearities for f_fi
        # so what's the kernel size
        self.conv = Conv_layer(config)
        # self.conv = resnet8_pretrained(config)
        self.activation = nn.GELU()

        self.add_pooling_layer = True
        self.MVLBert = MVLBert(config, add_pooling_layer=self.add_pooling_layer)

        transform = BertPredictionHeadTransform(config)
        linear = nn.Linear(config.hidden_size, config.result_num)
        self.final_mlp = nn.Sequential(
            # transform,
            nn.Dropout(config.hidden_dropout_prob, inplace=False),
            linear
        )
        # self.final_fc = nn.Linear(config.hidden_size, result_num)
        self.softmax = nn.Softmax(dim=-1)

        # print(self.state_dict())
        # for name, child in self._modules.items():
        #     print(name, child.__class__.__name__ if child is not None else 'None')
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, image, question, label, image_mask=None):
        """
        image: for convolution layer: (batch, channel, h, w)/ for linear patchs: (batch, obj_num, linear_patch_size)
        question: (batch, seq_len) id sequence of questions WITHOUT specail token such as [MAKS], [CLS], [SEP]
        label: (batch, 1)
        """
        image_feature = self.conv(image)  # (batch, obj_num, feature_dim)

        text_mask = (question > 0)
        text_idx = question
        batch_size = image_feature.shape[0]
        obj_num = image_feature.shape[1]
        if image_mask is None:
            image_mask = torch.ones((batch_size, obj_num), dtype=torch.bool, device=image_feature.device)

        MVLBert_output, pooler_output = self.MVLBert(text_idx=text_idx, text_mask=text_mask,
                                                     image_feature=image_feature, image_mask=image_mask)

        logits = self.final_mlp(pooler_output)
        prob = self.softmax(logits)
        return prob, logits


class MVLBertForPretraining(MVLBertPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.output_text_and_image_seperately = True
        self.conv = Conv_layer(config)
        # self.activation = nn.GELU()
        self.MVLBert = MVLBert(config, add_pooling_layer=True)

        self.MLM_head_seq2seq = BertOnlyMLMHead(config)
        self.MLM_head_bidir = BertOnlyMLMHead(config)
        self.ITM_mlp = nn.Linear(config.hidden_size, 2)

        # self.init_weights()
        # print(self.state_dict())
        # for name, child in self._modules.items():
        #     print(name, child.__class__.__name__ if child is not None else 'None')
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, image, caption_masked, caption_label, image_text_label, image_mask=None):
        """
        image: (batch, channel, h, w)
        caption_masked: (batch, seq_len) id sequence of caption WITH special tokens such as [MASK], [CLS], [SEP]
        caption_label: (batch, seq_len) gt of id sequence of caption for [MASK]
        image_text_label: (batch, 1)  indicate wehther image and text (caption) are paired, 1 for pair, 0 for un-pair
        image_mask:
        """
        image_feature = self.conv(image)  # (batch, obj_num, feature_dim)

        text_idx = caption_masked
        #  Mask to avoid performing attention on the padding token indices of the encoder input.
        text_mask = (text_idx > 0)
        batch_size = image_feature.shape[0]
        obj_num = image_feature.shape[1]
        if image_mask is None:
            image_mask = torch.ones((batch_size, obj_num), dtype=torch.bool, device=image_feature.device)

        _p = random.random()
        if _p < 0.5:
            seq2seq_mask = True
        else:
            seq2seq_mask = False

        MVLBert_text_output, _, pooled_output, _ \
            = self.MVLBert(text_idx, text_mask, image_feature, image_mask, seq2seq_mask=seq2seq_mask,
                           output_text_image_seperate=True)
        if seq2seq_mask is True:
            MLM_logits = self.MLM_head_seq2seq(MVLBert_text_output)
        else:
            MLM_logits = self.MLM_head_bidir(MVLBert_text_output)

        # print("MLM_logits.shape:", MLM_logits.shape, type(MLM_logits))
        # print("caption_label.shape:", caption_label.shape, type(caption_label))

        mlm_loss = torch.zeros((1, 1))
        itm_loss = None
        if self.config.MLM_task:
            mlm_loss = F.cross_entropy(MLM_logits.transpose(1, 2), caption_label, ignore_index=-100)
            # mlm_loss = F.cross_entropy(MLM_logits.view(-1, MLM_logits.shape[-1]), caption_label.view(-1).long())
        # print("ITM_logits.shape:", ITM_logits.shape)
        # print("image_text_label.shape:", image_text_label)

        if self.config.ITM_task:
            # print("calculate ITM loss")
            ITM_logits = self.ITM_mlp(pooled_output)
            itm_loss = F.cross_entropy(ITM_logits, image_text_label)

        return mlm_loss if itm_loss is None else mlm_loss.mean() + itm_loss.mean()


class MVLBertForRetrieval(MVLBertPretrainedModel):
    # given a text, the model need find the

    def __init__(self, config):
        super(MVLBertForRetrieval, self).__init__(config)
        self.config = config

        self.conv = Conv_layer(config)
        self.MVLBert = MVLBert(config, add_pooling_layer=True)

        # self.ITM_mlp = nn.Linear(config.hidden_size, 2)
        transform = BertPredictionHeadTransform(config)
        linear = nn.Linear(config.hidden_size, 2)
        self.final_mlp = nn.Sequential(
            transform,
            # nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            linear
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, caption, image_text_label=None, image_mask=None):
        """
        image: (batch, channel, h, w)
        caption_masked: (batch, seq_len) id sequence of caption WITH special token such as [CLS], [SEP]
        image_text_label: (batch, 1)  indicate wehther image and text (caption) are paired, 1 for pair, 0 for un-pair
        image_mask:
        """
        image_feature = self.conv(image)  # (batch, obj_num, feature_dim)
        text_idx = caption
        text_type = caption.new_zeros(caption.shape)
        #  Mask to avoid performing attention on the padding token indices of the encoder input.
        text_mask = (text_idx > 0)

        batch_size = image_feature.shape[0]
        obj_num = image_feature.shape[1]
        if image_mask is None:
            image_mask = torch.ones((batch_size, obj_num), dtype=torch.bool, device=image_feature.device)

        MVLBertoutput, pooled_output = self.MVLBert(text_idx=text_idx, text_mask=text_mask,
                                                    image_feature=image_feature, image_mask=image_mask)
        logits = self.final_mlp(pooled_output)

        if image_text_label == None:
            # prob -> sigmoid
            prob = self.softmax(logits)
            # prob = self.sigmoid(logits)
            return prob

        # binary_label = torch.zeros(ITM_logits.shape, dtype=torch.long)
        # binary_label[image_text_label] = 1
        # itm_loss = F.binary_cross_entropy_with_logits(ITM_logits, image_text_label)
        # itm_loss = F.cross_entropy(ITM_logits, image_text_label)
        return logits


class MVLBertForImageCaption(MVLBertPretrainedModel):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.config = config
        assert config.is_decoder, 'config.is_decoder should be True if you want to run image caption for testing'
        self.MVLBert = MVLBert(config, add_pooling_layer=True)
        self.conv = Conv_layer(config)
        self.tokenizer = tokenizer
        # self.MLM_new = BertOnlyMLMHead(config)
        self.MLM_head_seq2seq = BertOnlyMLMHead(config)

    def forward(self, image, caption, num_beams, learning_strategy, sample_mode='greedy'):
        """

        :param image:
        :param caption:
        :param num_beams:
        :param learning_strategy: ['unilm', 'normal']
        :param sample_mode:
        :return:
        """
        image_feature = self.conv(image)  # (batch, obj_num, feature_dim)
        batch_size = image_feature.shape[0]
        # text_idx = caption
        if num_beams > 1:
            _image_feature, _ = self._expand_inputs_for_generation(image_feature, expand_size=num_beams)
            beam_scorer = BeamSearchScorer(batch_size=batch_size, # max_length=self.config.max_length,
                                           num_beams=num_beams,
                                           device=image_feature.device)
            return self.beam_search(image_feature=_image_feature, beam_scorer=beam_scorer, input_ids=None,
                                    learning_strategy=learning_strategy, max_length=self.config.max_length,
                                    sample_mode=sample_mode, return_dict_in_generate=True)
        elif num_beams == 1:
            return self.greedy_search(image_feature=image_feature, input_ids=None, learning_strategy=learning_strategy,
                                      sample_mode=sample_mode, output_scores=True, return_dict_in_generate=True)
        else:
            return self.encode_forward(image_feature, caption, learning_strategy)

        # else:
        #     raise NotImplementedError("learning_strategy:", learning_strategy, "is not defined!" )

    def encode_forward(self, image_feature, caption, learning_strategy):
        # caption_label: batch, seq_len
        text_idx = caption
        text_mask = text_idx > 0
        batch_size = image_feature.shape[0]
        obj_num = image_feature.shape[1]
        image_mask = torch.ones((batch_size, obj_num), dtype=torch.bool, device=image_feature.device)

        text_output, image_output, _, sep_output = self.MVLBert(text_idx=text_idx, text_mask=text_mask,
                                                                image_feature=image_feature,
                                                                image_mask=image_mask, seq2seq_mask=True,
                                                                output_text_image_seperate=True)

        if learning_strategy == 'unilm':
            # hidden states of t1, t2, ..., tn, [END], shape is (batch, seq_len, hidden_size)
            # MLM_logits = self.MLM_new(text_output).transpose(1, 2)
            MLM_logits = self.MLM_head_seq2seq(text_output).transpose(1, 2)  # batch, vocab_size, seq_len
        elif learning_strategy == 'normal':
            # hidden states of [SEP], t1, t2, ..., tn, shape is (batch, seq_len, hidden_size)
            _text_output = torch.cat([sep_output[:, None], text_output[:, :-1]], dim=1)
            # print("_text_output:", _text_output.shape, "sep_output.shape:", sep_output.shape,"text_output.shape:", text_output.shape)
            MLM_logits = self.MLM_head_seq2seq(_text_output).transpose(1, 2)  # batch, vocab_size, seq_len
        else:
            raise NotImplementedError("learning_strategy:", learning_strategy,
                                      "is not implemented! Try 'unilm' or 'normal'.")

        return MLM_logits

    @staticmethod
    def _expand_inputs_for_generation(
            image_feature,
            expand_size=1,
            is_encoder_decoder=False,
            attention_mask=None,
            encoder_outputs=None,
            **model_kwargs):
        # expand the input image feature for beam_search
        expanded_return_idx = (
            torch.arange(image_feature.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(image_feature.device)
        )
        image_feature = image_feature.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return image_feature, model_kwargs

    def prepare_inputs_for_generation(self, image_feature, input_ids=None, past=None, text_mask=None,
                                      image_mask=None, learning_strategy=None, **model_kwargs):

        batch = image_feature.shape[0]
        if learning_strategy == 'unilm':
            mask_ids = torch.full((batch, 1), self.tokenizer.mask_token_id, dtype=torch.int64,device=image_feature.device)
            if input_ids is None:
                # for the first token.
                _input_ids = mask_ids
            else:
                _input_ids = torch.cat([input_ids, mask_ids], dim=-1)

            # cut decoder_input_ids if past is used, we only need the last generated token and [MASK] token
            if past is not None:
                _input_ids = _input_ids[:, -2:]

        elif learning_strategy == 'normal':
            _input_ids = input_ids
        else:
            raise NotImplementedError("learning_strategy:", learning_strategy,
                                      "is not implemented! Try 'unilm' or 'normal'.")

        if _input_ids is not None:
            text_mask = _input_ids > 0


        return {"text_idx": _input_ids, "text_mask": text_mask, 'image_feature': image_feature,'image_mask': image_mask,
                'past_key_values': past, 'use_cache': True, 'seq2seq_mask': True}

    # @staticmethod
    # def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
    #     # update past
    #     if "past_key_values" in outputs:
    #         model_kwargs["past"] = outputs.past_key_values
    #     elif "mems" in outputs:
    #         model_kwargs["past"] = outputs.mems
    #     elif "past_buckets_states" in outputs:
    #         model_kwargs["past"] = outputs.past_buckets_states
    #     else:
    #         model_kwargs["past"] = None
    #
    #     # update token_type_ids with last value
    #     if "token_type_ids" in model_kwargs:
    #         token_type_ids = model_kwargs["token_type_ids"]
    #         model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
    #
    #     # update attention mask
    #     if not is_encoder_decoder:
    #         if "attention_mask" in model_kwargs:
    #             attention_mask = model_kwargs["attention_mask"]
    #             model_kwargs["attention_mask"] = torch.cat(
    #                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    #             )
    #
    #     return model_kwargs

    def generate(self):
        pass

    def beam_search(
            self,
            image_feature,
            beam_scorer,
            input_ids=None,
            learning_strategy=None,
            logits_processor=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size = image_feature.shape[0]
        cur_len = 0

        assert (num_beams * batch_size == batch_beam_size), \
            "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=image_feature.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        obj_num = image_feature.shape[1]
        image_mask = torch.ones((batch_beam_size, obj_num), dtype=torch.bool, device=image_feature.device)

        past_key_values = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(image_feature, input_ids, past=past_key_values,
                                                              image_mask=image_mask,
                                                              learning_strategy=learning_strategy,
                                                              **model_kwargs)

            outputs, _ = self.MVLBert(**model_inputs)
            next_token_hidden = outputs.last_hidden_state[:, -1, :]  # (batch_size*num_beams, hidden)
            # next_token_logits = self.MLM_new(next_token_hidden)
            next_token_logits = self.MLM_head_seq2seq(next_token_hidden)


            if input_ids is None:
                if learning_strategy == 'unilm':
                    input_ids = model_inputs["text_idx"]
                    # _, next_tokens = torch.max(next_token_logits, dim=-1)  # batch,
                    # input_ids = next_tokens[:, None]
                else:
                    input_ids = torch.full((batch_beam_size, 1), self.tokenizer.sep_token_id, dtype=torch.int64, device=image_feature.device)


            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            # next_token_scores = logits_processor(input_ids, next_token_scores)

            # get all token log_scores for sequences in the beams.
            # for the first loop, the first beam of each sample in a batch is useful.
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores) # (batch_size * num_beams, vocab_size)
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            # next_token_scores, next_tokens: (batch_size, num_beams*2)
            # next_indices = next_tokens // vocab_size # old version of beam search
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")  # indices of beam
            next_tokens = next_tokens % vocab_size  # ids for the generated tokens

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]  # batch * num_beams
            beam_next_tokens = beam_outputs["next_beam_tokens"]  # batch * num_beams
            beam_idx = beam_outputs["next_beam_indices"]  # batch * num_beams


            # at each time step, past_key_values will increase by two, if the input_ids has two tokens,
            # one for the last generated token, another for [MASK], we only need the former, i.e. the generated token
            past_key_values = ()
            for past_key_value in outputs.past_key_values:
                # each past_key_value which is also a tuple, consists of two element, e.g. keys and values
                # both shapes are [batch_beam_size, num_attention_heads, seq_len, attention_head_size]
                key, value = past_key_value
                past_key_values += ((key[beam_idx, :, :-1], value[beam_idx, :, :-1]), )


            # past_key_values = None

            # at first, the input ids is set to [batch_beam_size] with all [MASK],
            # so here we just use the result of the beam_search to replace the [MASK] token.
            # In face, input_ids here is just generated sequences instead of input tokens.
            if cur_len == 0:
                input_ids = beam_next_tokens.unsqueeze(-1)
            else:
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            # # print(f"model_inputs['text_idx'], {model_inputs['text_idx'].shape}", end=' ')
            # # # s = model_inputs['text_idx'].shape
            # # shape = num_beams*model_inputs['text_idx'].shape[-1]
            # # input_tokens = self.tokenizer.convert_ids_to_tokens(model_inputs['text_idx'][-num_beams*1:, :].reshape(shape))
            # # print(f"input_tokens of 1: {input_tokens}", end=' ')
            #
            # # print(f"next_token_hidden.shape: {next_token_hidden.shape}", end=' ')
            # sample = 1
            # # print(f"next_tokens.shape, {next_tokens.shape}", end=' ') # batch, num_beams*2
            # cur_token_possible = self.tokenizer.convert_ids_to_tokens(next_tokens[sample])
            # print(f"current round generating: {sample} {cur_token_possible}", end=' ')
            #
            # sequence = self.tokenizer.convert_ids_to_tokens(input_ids[sample*num_beams]) if input_ids is not None else 0
            # print(f"sequence of sample: {sequence}", end=' ')
            # print()



        # (batch, seq_len) if we set 'num_beams_hyps_to_keep' = 1
        # if 'self.num_beam_hyps_to_keep` > 1, sequence_output will be (batch_size * self.num_beam_hyps_to_keep, seq_len)
        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, self.config.max_length, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        # if return_dict_in_generate:
        #     if not output_scores:
        #         sequence_outputs["sequence_scores"] = None
        #
        #     return BeamSearchDecoderOnlyOutput(
        #         sequences=sequence_outputs["sequences"],
        #         sequences_scores=sequence_outputs["sequence_scores"],
        #         scores=scores,
        #         attentions=decoder_attentions,
        #         hidden_states=decoder_hidden_states,
        #     )
        # else:
        return sequence_outputs["sequences"]

    @staticmethod
    def _init_sequence_length_for_generation(input_ids, max_length):
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    def greedy_search(
            self,
            image_feature,
            input_ids=None,
            learning_strategy=None,
            logits_processor=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            sample_mode='greedy',
            **model_kwargs,
    ):

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch = image_feature.shape[0]
        # init sequence length tensors
        # indicate whether sequences are ended
        unfinished_sequences = torch.ones(batch, dtype=torch.int64, device=image_feature.device)
        sequence_lengths = torch.ones(batch, dtype=torch.int64, device=image_feature.device).fill_(max_length)
        cur_len = 0

        obj_num = image_feature.shape[1]
        image_mask = torch.ones((batch, obj_num), dtype=torch.bool, device=image_feature.device)

        output_token_probs = []

        past_key_values = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(image_feature, input_ids, past=past_key_values,
                                                              image_mask=image_mask,
                                                              learning_strategy=learning_strategy,
                                                              **model_kwargs)
            outputs, _ = self.MVLBert(**model_inputs)
            # print("outputs.last_hidden_state.shape", outputs.last_hidden_state.shape)
            next_token_hidden = outputs.last_hidden_state[:, -1, :]

            # next_token_logits = self.MLM_new(next_token_hidden)
            next_token_logits = self.MLM_head_seq2seq(next_token_hidden)
            # at each time step, past_key_values will increase by two, if the input_ids has two token,
            # one for the last generated token, the another for [MASK]. We only need the former, the generated token
            # print(type(outputs.past_key_values), len(outputs.past_key_values))
            past_key_values = ()
            for past_key_value in outputs.past_key_values:
                # each past_key_value which is a tuple and consists of two element, key and value
                # past_key_values += ((past_key_value[0][:, :, :image_len], past_key_value[1][:, :, :image_len]),)
                past_key_values += ((past_key_value[0][:, :, :-1], past_key_value[1][:, :, :-1]), )

            if sample_mode == 'greedy':
                next_token_scores, next_tokens = torch.max(next_token_logits, dim=-1) # batch,
            elif sample_mode == 'sample':
                # next_token_logits.squeeze_(1)
                prediction_probs = F.softmax(next_token_logits, dim=-1).detach()
                next_tokens = torch.multinomial(prediction_probs, num_samples=1,
                                                replacement=True)

                next_token_scores = torch.gather(F.log_softmax(next_token_logits, dim=-1), 1,
                                                 next_tokens)  # this should be logprobs
                next_tokens = next_tokens.squeeze(1)
            else:
                print("sample mode error!")
                return

            scores += (next_token_logits.unsqueeze(2),)
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            # print("input_ids.shape:", input_ids[:, :-1].shape, "next_tokens.shape:", next_tokens.shape)
            if input_ids is None:
                input_ids = next_tokens[:, None]
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )
            # # update model kwargs
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break


            # increase cur_len
            cur_len = cur_len + 1

            # input_tokens = self.tokenizer.convert_ids_to_tokens(model_inputs['text_idx'][0])
            # print(f"input_tokens of 0: {input_tokens}", end=' ')
            # # print(f"next_token_hidden.shape: {next_token_hidden.shape}", end=' ')
            # cur_token = self.tokenizer.convert_ids_to_tokens(next_tokens)
            # print(f"generate {cur_token}", end=' ')
            # sequence = self.tokenizer.convert_ids_to_tokens(input_ids[0]) if input_ids is not None else 0
            # print(f"sequence of 0: {sequence}", end=' ')
            # print()

            output_token_probs.append(next_token_scores)

        # if return_dict_in_generate:
        #     return GreedySearchDecoderOnlyOutput(
        #         sequences=input_ids,
        #         scores=scores,
        #         attentions=decoder_attentions,
        #         hidden_states=decoder_hidden_states,
        #     )
        # else:
        # logits_all_tokens = torch.cat(scores, dim=-1)

        # while logits_all_tokens.shape[-1] < max_length:

        return input_ids, torch.cat(output_token_probs, dim=-1),

    @staticmethod
    def _update_seq_length_for_generation(
            sequence_lengths: torch.LongTensor,
            unfinished_sequences: torch.LongTensor,
            cur_len: int,
            is_eos_in_next_token: torch.BoolTensor,
    ):
        # check if sentence is not finished yet
        is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

        # update sentence length
        sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
        unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
        return sequence_lengths, unfinished_sequences



