# SPDX-License-Identifier: Apache-2.0
import torch


class GPTQUtils:

    def __init__(self, bits=4, blocksize=128):
        super(GPTQUtils, self).__init__()  # noqa: UP008
        self.bits = bits
        self.blocksize = blocksize

    def convert_idx(self, g_idx, k):
        ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
        groups = k // self.blocksize
        remainder = k % self.blocksize
        g_idx_2 = g_idx * self.blocksize
        if remainder > 0:
            g_idx_2[g_idx == groups] += torch.arange(remainder).to(
                g_idx.device)
        arrange_tensor = torch.arange(self.blocksize).to(g_idx.device)
        for i in range(groups):
            g_idx_2[g_idx == i] += arrange_tensor
        ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
        return ret_idx.to(torch.int32)

    def unpack_weight(self, qweight_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (torch.tensor(list(range(0, s32_bits, self.bits)),
                           dtype=torch.int32).unsqueeze(0).to(
                               qweight_int32.device))
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight_int32, 1).expand(-1, 32 // self.bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)

        return weight

    def unpack_zp(self, qzeros_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (torch.tensor(list(range(0, s32_bits, self.bits)),
                           dtype=torch.int32).unsqueeze(0).to(
                               qzeros_int32.device))
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros_int32, 2).expand(-1, -1, 32 // self.bits),
            wf.unsqueeze(0),
        ).to(torch.int8)
        torch.bitwise_and(zeros, (2**self.bits) - 1, out=zeros)

        return zeros

    def pack(self, qweight_int8):
        i = 0
        row = 0
        qweight_int32_shape = (
            qweight_int8.shape[0] // 32 * self.bits,
            qweight_int8.shape[1],
        )
        qweight_int32 = torch.zeros(qweight_int32_shape,
                                    dtype=torch.int32,
                                    device=qweight_int8.device)

        while row < qweight_int32.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight_int32[row] |= qweight_int8[j].to(
                        torch.int32) << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        return qweight_int32

    def shuffle(self, qweight_int32, g_idx):
        k = qweight_int32.shape[0] * 8
        g_idx4kernel = self.convert_idx(g_idx, k).to(qweight_int32.device)
        qweight_int8 = self.unpack_weight(qweight_int32)
        qweight_int8 = qweight_int8.reshape(-1, qweight_int8.shape[-1])
        qweight_int8_shuffled = qweight_int8[g_idx4kernel, :]
        qweight_int32_shuffled = self.pack(qweight_int8_shuffled)
        return qweight_int32_shuffled, g_idx4kernel


class AWQUtils:
    AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
    REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    @classmethod
    def pack(cls, imatrix: torch.Tensor, direction: str = "column"):
        """
        Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            direction (str): direction of packing, either "column" or "row"
        Returns:
            qmatrix (torch.Tensor): packed matrix of integers
        """
        shifts = torch.arange(0,
                              32,
                              4,
                              dtype=torch.int32,
                              device=imatrix.device)

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        if direction == "column":
            imatrix = imatrix.view(-1, imatrix.shape[1] // (32 // 4),
                                   (32 // 4))
            qmatrix = torch.bitwise_left_shift(imatrix,
                                               shifts[None,
                                                      None, :]).sum(dim=-1)

        elif direction == "row":
            imatrix = imatrix.view(imatrix.shape[0] // (32 // 4), (32 // 4),
                                   -1)
            qmatrix = torch.bitwise_left_shift(imatrix,
                                               shifts[None, :,
                                                      None]).sum(dim=1)

        qmatrix = qmatrix.to(torch.int32)

        return qmatrix

    @classmethod
    def unpack(cls, qmatrix: torch.Tensor, direction: str = "column"):
        """
        Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.
        Args:
            qmatrix (torch.Tensor): matrix of packed integers
            direction (str): direction of unpacking, either "column" or "row"
        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        shifts = torch.arange(0, 32, 4, device=qmatrix.device)

        if direction == "column":
            imatrix = torch.bitwise_right_shift(qmatrix[:, :, None],
                                                shifts[None, None, :]).view(
                                                    qmatrix.shape[0], -1)

        elif direction == "row":
            imatrix = torch.bitwise_right_shift(qmatrix[:, None, :],
                                                shifts[None, :, None]).view(
                                                    -1, qmatrix.shape[-1])

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        return imatrix

    @classmethod
    def apply_order(
        cls,
        imatrix: torch.Tensor,
        direction: str = "column",
        order: list[int] = None,
    ):
        """
        Applies the order to a 4-bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            direction (str): direction of applying order, "column" or "row"
            order (List[int]): order to apply, default is AWQ_PACK_ORDER
        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        if direction == "column":
            imatrix = imatrix.view(-1, (32 // 4))[:, order].view(imatrix.shape)
        elif direction == "row":
            imatrix = imatrix.view((32 // 4), -1)[order, :].view(imatrix.shape)

        return imatrix

    @classmethod
    def repack(cls, qweight, qzeros):
        # awq uses column packing for both weights and zeros
        izeros = cls.unpack(qzeros, direction="column")
        iweights = cls.unpack(qweight, direction="column")

        # Reverse the order of the iweight and izeros tensors
        izeros = cls.apply_order(izeros,
                                 direction="column",
                                 order=cls.REVERSE_AWQ_PACK_ORDER)
        iweights = cls.apply_order(iweights,
                                   direction="column",
                                   order=cls.REVERSE_AWQ_PACK_ORDER)

        # exllama uses row packing for weights and column packing for zeros
        qzeros = cls.pack(izeros, direction="column")
        qweight = cls.pack(iweights, direction="row")

        return qweight, qzeros


def transpose_onednn_woq_format(layer: torch.nn.Module,
                                method: str,
                                is_sym: bool = True):
    # The oneDNN int4 GEMM has the following requirements:
    # - Weights need to be contiguous along the 'k' dimension,
    #   but the shape should remain (k, n/8).
    # - Scales remains unchanged.
    # - Zero-point is a scalar value of 8 in symmetric (symm) scenarios,
    #   allowing oneDNN to broadcast it.
    # - Zero-point remains unchanged in asymmetric (asymm) scenarios.
    reshaped_tensor = layer.qweight.transpose(0,
                                              1).contiguous().transpose(0, 1)
    layer.qweight.as_strided_(reshaped_tensor.shape, reshaped_tensor.stride())
    layer.qweight.copy_(reshaped_tensor)
    layer.scales.data = layer.scales.contiguous()
    if method == "gptq":
        if is_sym:
            qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        else:
            qzeros = layer.qzeros + 0x11111111
        layer.qzeros.copy_(qzeros)


def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
    # Dequantize the weight from int4 to scales data type
    gptq_utils = GPTQUtils(bits=4, blocksize=group_size)
    weight = gptq_utils.unpack_weight(qweight)
    if len(weight.shape) > 2:
        weight = weight.reshape(-1, weight.shape[-1])
    infeatures = weight.shape[0]
    if g_idx is None:
        g_idx = torch.tensor(
            [i // group_size for i in range(infeatures)],
            dtype=torch.int32,
        )
    if qzeros is None:
        return (weight - 8) * scales[g_idx]
    else:
        gptq_zeros = gptq_utils.unpack_zp(qzeros)
        gptq_zeros = gptq_zeros.reshape(scales.shape)
        return (weight - gptq_zeros[g_idx]) * scales[g_idx]
