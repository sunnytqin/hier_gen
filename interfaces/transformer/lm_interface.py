import torch
import torch.nn
from typing import Dict, Tuple
from ..model_interface import ModelInterface
from ..encoder_decoder import EncoderDecoderResult
from models.transformer_enc_dec import TransformerResult
from models.encoder_decoder import add_eos
import layers


class TransformerLMInterface(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        super(TransformerLMInterface, self).__init__()
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True, decl_lm_only=False,
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat(
            [sos_tensor, data["in"]],
            dim=0,
        ).transpose(0, 1)

        out_data = add_eos(
            data["in"], data["in_len"], self.model.encoder_eos
        ).transpose(0, 1)

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        res = self.model(inp_data, in_len)

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(inp_data.shape[1], in_len).transpose(
            0, 1
        )

        # create two additional len_mask, one for the input and one for the output
        # len_mask =  bs x seq_len
        len_mask_first_half = ~self.model.generate_len_mask(inp_data.shape[1], data["prefix_len"] + 1).transpose(
            0, 1
        ) # I think here should be "+1" because now the second half includes "quest" and "decl", and in the forward pass, it predicts next token after
        len_mask_second_half = torch.bitwise_xor(len_mask_first_half, len_mask)

        if decl_lm_only: # do not model the first half of the sentence for all quest    
            quest_sent_flag = torch.sum(inp_data == 37, dim=1)
            len_mask_quest_second_half = (len_mask_second_half.clone() * quest_sent_flag).bool()
            decl_sent_flag = torch.sum(inp_data == 15, dim=1)
            # len_mask_decl_second_half = (len_mask_second_half.clone() * decl_sent_flag).bool()
            len_mask_decl_full = (len_mask.clone() * decl_sent_flag).bool()

            # len_mask = torch.bitwise_or(len_mask_quest_second_half, len_mask_decl_second_half)
            len_mask = torch.bitwise_or(len_mask_quest_second_half, len_mask_decl_full)

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
        loss_first_half = self.loss(res, out_data.transpose(0, 1), len_mask_first_half, normalize)
        loss_second_half = self.loss(res, out_data.transpose(0, 1), len_mask_second_half, normalize)

        loss_dict = {
            "loss": loss, 
            "loss_first_half": loss_first_half, # task agnostic
            "loss_second_half": loss_second_half # task agnostic
        }


        if "reg" in res:
            return EncoderDecoderResult(res.data, res.length, loss, res.reg, None)
        else:
            return EncoderDecoderResult(res.data, res.length, loss, None, loss_dict)
