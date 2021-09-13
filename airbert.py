import torch
from typing import Dict, List

from vilbert.vilbert import (
    BertModel as ViLBertModel,
    BertPreTrainedModel as PreTrainedModel,
    BertPreTrainingHeads as ViLBertPreTrainingHeads,
    BertConfig as ViLBertConfig,
)

BERT_CONFIG_FACTORY = {
    "vilbert": ViLBertConfig,
}

BERT_MODEL_FACTORY = {
    "vilbert": ViLBertModel,
}
CLS_MODEL_FACTORY = {
    "vilbert": ViLBertPreTrainingHeads,
}


class Airbert(PreTrainedModel):
    def __init__(self, config, dropout_prob=0.1):
        super().__init__(config)

        # vision and language processing streams
        self.bert = BERT_MODEL_FACTORY[config.model_name](config)

        # pre-training heads
        self.cls = CLS_MODEL_FACTORY[config.model_name](
            config, self.bert.embeddings.word_embeddings.weight
        )

        # word-level prediction
        voc_size = self.bert.embeddings.word_embeddings.num_embeddings
        # self.highlighter = torch.nn.Linear(voc_size, 1)
        self.cat_highlight = config.cat_highlight
        self.no_ranking = config.no_ranking
        self.masked_vision = config.masked_vision
        self.masked_language = config.masked_language

        # path selection head
        bi_hidden_size = (
            config.bi_hidden_size
            if config.model_name == "vilbert"
            else config.hidden_size
        )
        # if self.cat_highlight:
        #     self.vil_logit2 = torch.nn.Linear(bi_hidden_size + voc_size, 1)
        # else:
        self.vil_logit = torch.nn.Linear(bi_hidden_size, 1)

        # misc
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fusion_method = (
            config.fusion_method if config.model_name != "oscar" else None
        )

        self.apply(self.init_bert_weights)

    def forward(
        self,
        instr_tokens,
        image_features,
        image_locations,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        highlight_tokens=None,
    ) -> Dict[str, torch.Tensor]:
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            _,
        ) = self.bert(
            input_txt=instr_tokens,
            input_imgs=image_features,
            image_loc=image_locations,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            co_attention_mask=co_attention_mask,
            output_all_encoded_layers=False,
        )

        linguistic_prediction, vision_prediction, _ = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        if self.config.model_name == "oscar":
            pooled_output = pooled_output_t
        elif self.fusion_method == "sum":
            pooled_output = pooled_output_t + pooled_output_v
        elif self.fusion_method == "mul":
            pooled_output = pooled_output_t * pooled_output_v
        else:
            assert False

        pooled_output = self.dropout(pooled_output)

        outputs = {}

        # if highlight_tokens is not None and highlight_tokens.numel() > 0:
        #     highlight_logit = (
        #         linguistic_prediction * highlight_tokens.unsqueeze(2).float()
        #     ).sum(1)
        #     highlight_prediction = self.highlighter(highlight_logit)

        # else:
        highlight_prediction = None
        highlight_logit = None

        # if self.cat_highlight:
        #     pooled_output = torch.cat([pooled_output, highlight_logit], dim=1)  # type: ignore
        #     vil_logit = self.vil_logit2(pooled_output)
        # else:

        # When using a DDP over multiple machines, PyTorch is complaining about unused outputs
        if not self.no_ranking:
            outputs["ranking"] = self.vil_logit(pooled_output)

        if self.masked_vision:
            outputs["vision"] = vision_prediction

        if self.masked_language:
            outputs["language"] = linguistic_prediction

        return outputs


class VLNOSCAR(OscarPreTrainedModel):
    def __init__(self, config, dropout_prob=0.1):
        super().__init__(config)
        # import ipdb

        # ipdb.set_trace()

        # vision and language processing streams
        self.bert = OscarModel(config)

        # pre-training heads
        self.cls = OscarPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )

        # word-level prediction
        voc_size = self.bert.embeddings.word_embeddings.num_embeddings
        self.highlighter = torch.nn.Linear(voc_size, 1)
        self.cat_highlight = config.cat_highlight

        # path selection head
        bi_hidden_size = config.hidden_size
        if self.cat_highlight:
            self.vil_logit2 = torch.nn.Linear(bi_hidden_size + voc_size, 1)
        else:
            self.vil_logit = torch.nn.Linear(bi_hidden_size, 1)

        # misc
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.apply(self.init_weights)

    def forward(
        self,
        instr_tokens,
        image_features,
        image_locations,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        highlight_tokens=None,
    ) -> Dict[str, torch.Tensor]:
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            _,
        ) = self.bert(
            input_txt=instr_tokens,
            input_imgs=image_features,
            image_loc=image_locations,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            co_attention_mask=co_attention_mask,
            output_all_encoded_layers=False,
        )

        linguistic_prediction, vision_prediction, _ = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        pooled_output = self.dropout(pooled_output_t)

        outputs = {}

        if highlight_tokens is not None and highlight_tokens.numel() > 0:
            highlight_logit = (
                linguistic_prediction * highlight_tokens.unsqueeze(2).float()
            ).sum(1)
            highlight_prediction = self.highlighter(highlight_logit)

        else:
            highlight_prediction = None
            highlight_logit = None

        if self.cat_highlight:
            pooled_output = torch.cat([pooled_output, highlight_logit], dim=1)  # type: ignore
            vil_logit = self.vil_logit2(pooled_output)
        else:
            vil_logit = self.vil_logit(pooled_output)

        return {
            "action": vil_logit,
            "vision": vision_prediction,
            "language": linguistic_prediction,
            # highlight_prediction,
        }
