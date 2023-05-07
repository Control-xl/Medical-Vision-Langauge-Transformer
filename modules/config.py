from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertTokenizerFast

class MVLBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type_vocab_size = 3
        self.MLM_task = kwargs.pop('MLM_task', True)
        self.ITM_task = kwargs.pop('ITM_task', True)
        self.conv = kwargs.pop('conv', 'resnet101') # ['resnet101', 'linear', '
        # self.output_text_and_image_seperately = False
        self.result_num = 224
        self.lr = 4e-5 # 6*1e-5
        self.max_length = kwargs.pop('max_length', 40)
        # self.eos_token_id = None
        # self.cls_token_id = None
        # self.sep_token_id = None
        self.mask_token_id = None
        self.attention_probs_dropout_prob = 0.0
        self.hidden_dropout_prob = 0.0

    def update_special_tokens(self, tokenizer):
        self.eos_token_id, self.cls_token_id, self.sep_token_id, self.mask_token_id = tokenizer.convert_tokens_to_ids(['[END]', '[CLS]', '[SEP]', '[MASK]'])
        self.vocab_size = len(tokenizer)
        print("eos_token_id:", self.eos_token_id, "cls_token_id:", self.cls_token_id, "sep_token_id:", self.sep_token_id, 'mask_token_id:', self.mask_token_id)
        return

class MVLBertConfigforVQA(MVLBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_vocab_size = 3
        self.MLM_task = True
        self.ITM_task = True
        self.result_num = 224
        self.lr = 4e-5 # 6*1e-5
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1


class MVLBertPretrainConfig(MVLBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_vocab_size = 3
        self.MLM_task = True
        self.ITM_task = False
        self.max_length = 150
        self.lr = 4e-5 # 6*1e-5
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1


class MVLBertRetrieval(MVLBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_vocab_size = 3
        self.ITM_task = True
        self.lr = 1e-6 # 6*1e-5
        self.max_length = 80
        self.attention_probs_dropout_prob = 0.1



class MVLBertConfigForImageCaption(MVLBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_vocab_size = 3
        self.lr = 1e-5
        self.max_length = 80
        self.is_decoder = True
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1




if __name__ == '__main__':
    a = MVLBertConfigforVQA.from_pretrained('../checkpoints/resnet50-bert-base')
    from utils import print_obj
    print_obj(a)


