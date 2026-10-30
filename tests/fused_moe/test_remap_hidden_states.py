# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import seed_everything

DEVICE = "xpu"
NUM_ROWS = [32, 1024]
HIDDEN_SIZE = [128]
TOTAL_EXPERTS_NUM = [32, 128, 2560]
TOP_KS = [1, 3, 8, 12, 16]
RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "fp8": (torch.float8_e4m3fn, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}

LOCAL_EXPERTS_NUM = [3, 8, 11]
EP_RANK = [0, 1, 2, 3]
EP_SIZE = [4]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_rows": [32],
        "hidden_size": [128],
        "total_experts_num": [16],
        "topk": [1],
    },
}


def ref_remap_hidden_states(hidden_states, scales, remapped_hidden_states,
                            remapped_scales, expert_map,
                            rows_per_expert,
                            unpermuted_row_to_permuted_row, topk_ids,
                            total_experts_num, local_experts_num):
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids]
    else:
        local_topk_ids = topk_ids

    valid_mask = local_topk_ids >= 0
    valid_tensor = local_topk_ids[valid_mask]

    frequencies = torch.bincount(valid_tensor.flatten(),
                                 minlength=local_experts_num)
    prefix = torch.cat([
        torch.zeros(1, dtype=frequencies.dtype, device=frequencies.device),
        torch.cumsum(frequencies, dim=0)
    ])

    rows_per_expert.copy_(frequencies.to(torch.int32))

    expert_local_offset = torch.zeros((local_experts_num, ),
                                      dtype=torch.int32,
                                      device=local_topk_ids.device)
    num_rows = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    if scales is not None and scales.dtype is torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)

    for i in range(num_rows):
        for j in range(topk):
            selected_expert = local_topk_ids[i, j].item()
            if selected_expert == -1:
                unpermuted_row_to_permuted_row[i, j] = -1
                continue
            first_token_offset_offset = prefix[
                selected_expert].item()
            offset = expert_local_offset[selected_expert]
            remapped_hidden_states[first_token_offset_offset +
                                   offset] = hidden_states[i]
            if scales is not None:
                remapped_scales[first_token_offset_offset + offset] = scales[i]
            unpermuted_row_to_permuted_row[
                i, j] = first_token_offset_offset + offset
            expert_local_offset[selected_expert] += 1


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("total_experts_num", TOTAL_EXPERTS_NUM)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("has_expert_map", [True, False])
@pytest.mark.parametrize("recipe",
                         ["bf16", "fp16", "fp8", "mxfp8", "mxfp4", "fp8block"])
def test_remap_hidden_states(num_rows, hidden_size, total_experts_num, topk,
                             has_expert_map, recipe, *, ids_dtype=None,
                             routing="random"):
    seed_everything(7)

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))

    if has_expert_map:
        local_experts_num = total_experts_num // 2
    else:
        local_experts_num = total_experts_num

    if data_dtype in [torch.bfloat16, torch.float16]:
        hidden_states = torch.randn((num_rows, hidden_size),
                                    dtype=data_dtype,
                                    device=DEVICE)
        scales = None
    elif data_dtype is torch.float8_e4m3fn:
        hidden_states_fp32 = torch.randn((num_rows, hidden_size),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.float8_e4m3fn)
        if recipe == "fp8block":
            block_k = 128
            scales = torch.randn((num_rows, hidden_size // block_k),
                                 device=DEVICE,
                                 dtype=torch.float32)
        elif recipe == "mxfp8":
            block_k = 16
            scales = torch.randint(1,
                                   256, (num_rows, hidden_size // block_k),
                                   device=DEVICE,
                                   dtype=torch.uint8).view(
                                       torch.float8_e8m0fnu)
        elif recipe == "fp8":
            scales = None
    elif data_dtype is torch.float4_e2m1fn_x2:
        block_k = 16  # two input elem in a 8bit
        hidden_states_fp32 = torch.randn((num_rows, hidden_size // 2),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.uint8).view(
            torch.float4_e2m1fn_x2)
        scales = torch.randint(1,
                               256, (num_rows, hidden_size // 2 // block_k),
                               device=DEVICE,
                               dtype=torch.uint8).view(torch.float8_e8m0fnu)

    remapped_hidden_states = torch.empty_like(hidden_states).repeat_interleave(
        topk, dim=0)
    remapped_scales = None
    if scale_dtype is not None:
        remapped_scales = torch.empty_like(scales).repeat_interleave(topk,
                                                                     dim=0)
    rows_per_expert = torch.zeros((local_experts_num),
                                            dtype=torch.int32,
                                            device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty((num_rows, topk),
                                                 dtype=torch.int32,
                                                 device=DEVICE)

    expert_map = None
    if has_expert_map:
        expert_map = torch.full((total_experts_num, ),
                                -1,
                                dtype=torch.int64,
                                device=DEVICE)
        expert_map[torch.randperm(
            total_experts_num,
            device=DEVICE)[:local_experts_num]] = torch.randperm(
                local_experts_num, device=DEVICE)
        expert_map = expert_map.to(torch.int32)

    scores = torch.randn((num_rows, total_experts_num),
                         device=DEVICE,
                         dtype=torch.float32)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    if routing == "all_nonlocal":
        assert expert_map is not None
        nonlocal_ids = torch.where(expert_map < 0)[0]
        topk_ids = nonlocal_ids[:topk].expand(num_rows, topk).clone()
    elif routing == "concentrated":
        topk_ids = torch.arange(topk, device=DEVICE).expand(num_rows,
                                                            topk).clone()
    is_topk_ids_32 = random.randint(0, 1)
    if is_topk_ids_32:
        topk_ids = topk_ids.to(torch.int32)
    else:
        topk_ids = topk_ids.to(torch.int64)
    if ids_dtype is not None:
        topk_ids = topk_ids.to(ids_dtype)

    ref_remapped_hidden_states = remapped_hidden_states.clone()
    ref_rows_per_expert = rows_per_expert.clone()
    ref_unpermuted_row_to_permuted_row = unpermuted_row_to_permuted_row.clone()
    ref_remapped_scales = None
    if scale_dtype is not None:
        ref_remapped_scales = remapped_scales.clone()
        if scales.dtype is torch.float8_e8m0fnu:
            ref_remapped_scales = ref_remapped_scales.view(torch.uint8)

    ref_remap_hidden_states(hidden_states, scales, ref_remapped_hidden_states,
                            ref_remapped_scales, expert_map,
                            ref_rows_per_expert,
                            ref_unpermuted_row_to_permuted_row, topk_ids,
                            total_experts_num, local_experts_num)

    torch.ops._moe_C.remap_hidden_states(
        hidden_states, scales, remapped_hidden_states, remapped_scales,
        expert_map, rows_per_expert, unpermuted_row_to_permuted_row,
        topk_ids, total_experts_num, local_experts_num)

    if data_dtype is torch.float4_e2m1fn_x2:
        remapped_hidden_states = remapped_hidden_states.view(torch.uint8)
        ref_remapped_hidden_states = ref_remapped_hidden_states.view(
            torch.uint8)

    valid = ref_unpermuted_row_to_permuted_row >= 0
    assert torch.all(unpermuted_row_to_permuted_row[~valid] == -1)
    valid_rows = unpermuted_row_to_permuted_row[valid]
    ref_valid_rows = ref_unpermuted_row_to_permuted_row[valid]
    assert torch.all(valid_rows >= 0)
    assert torch.all(valid_rows < rows_per_expert.sum())
    assert torch.unique(valid_rows).numel() == valid_rows.numel()
    unpermuted_hidden_states = remapped_hidden_states[valid_rows]
    ref_unpermuted_hidden_states = ref_remapped_hidden_states[ref_valid_rows]

    torch.testing.assert_close(unpermuted_hidden_states,
                               ref_unpermuted_hidden_states,
                               rtol=0,
                               atol=0,
                               equal_nan=True)
    torch.testing.assert_close(ref_rows_per_expert,
                               rows_per_expert,
                               rtol=0,
                               atol=0)
    if scale_dtype is not None:
        unpermuted_scales = remapped_scales[valid_rows]
        ref_unpermuted_scales = ref_remapped_scales[ref_valid_rows]
        if unpermuted_scales.dtype is torch.float8_e8m0fnu:
            unpermuted_scales = unpermuted_scales.view(torch.uint8)
            ref_unpermuted_scales = ref_unpermuted_scales.view(torch.uint8)
        torch.testing.assert_close(unpermuted_scales,
                                   ref_unpermuted_scales,
                                   rtol=0,
                                   atol=0,
                                   equal_nan=True)


@pytest.mark.parametrize("recipe,num_rows,hidden_size,topk,routing,ids_dtype", [
    ("bf16", 1, 16, 1, "random", torch.int32),
    ("fp16", 32, 128, 8, "random", torch.int64),
    ("bf16", 512, 4224, 10, "random", torch.int32),
    ("fp16", 512, 4224, 10, "random", torch.int64),
    ("bf16", 513, 4224, 10, "random", torch.int64),
    ("fp16", 513, 4224, 10, "random", torch.int32),
    ("fp8", 513, 4224, 10, "random", torch.int64),
    ("mxfp8", 513, 4224, 10, "random", torch.int32),
    ("mxfp4", 513, 4224, 10, "random", torch.int64),
    ("fp8block", 513, 4224, 10, "random", torch.int32),
    ("bf16", 1024, 4096, 8, "random", torch.int32),
    ("fp16", 1025, 4128, 10, "random", torch.int64),
    ("mxfp8", 1025, 128, 1, "random", torch.int32),
    ("bf16", 513, 4224, 10, "all_nonlocal", torch.int32),
    ("fp16", 1025, 128, 8, "all_nonlocal", torch.int64),
    ("bf16", 1025, 4096, 1, "concentrated", torch.int64),
])
def test_remap_hidden_states_copy_edges(recipe, num_rows, hidden_size, topk,
                                       routing, ids_dtype):
    # Use representative combinations, not a full Cartesian product.
    # 4224 is scale-block aligned with a copy tail for every recipe.
    test_remap_hidden_states(
        num_rows, hidden_size, 32, topk,
        routing != "concentrated", recipe, ids_dtype=ids_dtype, routing=routing)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_remap_ignores_negative_expert_ids(id_dtype):
    """A padded route must not read the entry before the expert map."""
    hidden = torch.ones((2, 128), dtype=torch.bfloat16, device=DEVICE)
    # Make an erroneous expert_map[-1] access deterministic.
    map_storage = torch.tensor([0, 0, -1], dtype=torch.int32,
                               device=DEVICE)
    expert_map = map_storage[1:]
    assert expert_map.storage_offset() == 1
    assert (expert_map.untyped_storage().data_ptr() ==
            map_storage.untyped_storage().data_ptr())
    ids = torch.tensor([[0, 1], [-1, -1]], dtype=id_dtype, device=DEVICE)
    remapped = torch.empty((4, 128), dtype=hidden.dtype, device=DEVICE)
    counts = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    mapping = torch.empty((2, 2), dtype=torch.int32, device=DEVICE)

    torch.ops._moe_C.remap_hidden_states(
        hidden, None, remapped, None, expert_map, counts, mapping, ids, 2, 1)

    assert counts.tolist() == [1]
    assert mapping.cpu().tolist() == [[0, -1], [-1, -1]]
    torch.testing.assert_close(remapped[0], hidden[0], rtol=0, atol=0)


@pytest.mark.parametrize("num_rows", [262144])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("total_experts_num", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("has_expert_map", [False])
@pytest.mark.parametrize("recipe",
                         ["bf16", "fp16", "mxfp8", "mxfp4", "fp8block"])
def test_remap_hidden_states_overflow(num_rows, hidden_size, total_experts_num,
                                      topk, has_expert_map, recipe):
    seed_everything(7)

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))

    if has_expert_map:
        local_experts_num = total_experts_num // 2
    else:
        local_experts_num = total_experts_num

    if data_dtype in [torch.bfloat16, torch.float16]:
        hidden_states = torch.randn((num_rows, hidden_size),
                                    dtype=data_dtype,
                                    device=DEVICE)
        scales = None
    elif data_dtype is torch.float8_e4m3fn:
        hidden_states_fp32 = torch.randn((num_rows, hidden_size),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.float8_e4m3fn)
        if recipe == "fp8block":
            block_k = 128
            scales = torch.randn((num_rows, hidden_size // block_k),
                                 device=DEVICE,
                                 dtype=torch.float32)
        elif recipe == "mxfp8":
            block_k = 16
            scales = torch.randint(1,
                                   256, (num_rows, hidden_size // block_k),
                                   device=DEVICE,
                                   dtype=torch.uint8).view(
                                       torch.float8_e8m0fnu)
    elif data_dtype is torch.float4_e2m1fn_x2:
        block_k = 16  # two input elem in a 8bit
        hidden_states_fp32 = torch.randn((num_rows, hidden_size // 2),
                                         dtype=torch.float32,
                                         device=DEVICE)
        hidden_states = hidden_states_fp32.to(torch.uint8).view(
            torch.float4_e2m1fn_x2)
        scales = torch.randint(1,
                               256, (num_rows, hidden_size // 2 // block_k),
                               device=DEVICE,
                               dtype=torch.uint8).view(torch.float8_e8m0fnu)

    remapped_hidden_states = torch.empty_like(hidden_states).repeat_interleave(
        topk, dim=0)
    remapped_scales = None
    if scale_dtype is not None:
        remapped_scales = torch.empty_like(scales).repeat_interleave(topk,
                                                                     dim=0)
    rows_per_expert = torch.zeros((local_experts_num),
                                            dtype=torch.int32,
                                            device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty((num_rows, topk),
                                                 dtype=torch.int32,
                                                 device=DEVICE)

    expert_map = None
    if has_expert_map:
        expert_map = torch.full((total_experts_num, ),
                                -1,
                                dtype=torch.int64,
                                device=DEVICE)
        expert_map[torch.randperm(
            total_experts_num,
            device=DEVICE)[:local_experts_num]] = torch.randperm(
                local_experts_num, device=DEVICE)
        expert_map = expert_map.to(torch.int32)

    scores = torch.randn((num_rows, total_experts_num),
                         device=DEVICE,
                         dtype=torch.float32)
    _, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int32)

    torch.ops._moe_C.remap_hidden_states(
        hidden_states, scales, remapped_hidden_states, remapped_scales,
        expert_map, rows_per_expert, unpermuted_row_to_permuted_row,
        topk_ids, total_experts_num, local_experts_num)

    # Keep the reference checks on the host.
    def _cpu(x):
        return None if x is None else x.cpu()

    mapping = unpermuted_row_to_permuted_row.cpu().to(torch.int64)
    counts_k = rows_per_expert.cpu()
    hidden_states, scales, expert_map, topk_ids = map(
        _cpu, (hidden_states, scales, expert_map, topk_ids))
    remapped_hidden_states, remapped_scales = map(
        _cpu, (remapped_hidden_states, remapped_scales))
    local_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
    valid = local_ids >= 0
    counts = torch.bincount(local_ids[valid], minlength=local_experts_num)
    torch.testing.assert_close(counts_k.to(torch.int64), counts)
    assert torch.all(mapping[~valid] == -1)
    torch.testing.assert_close(
        mapping[valid].sort().values, torch.arange(int(counts.sum())))
    prefix = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    assert torch.all(mapping[valid] >= prefix[local_ids[valid]])
    assert torch.all(mapping[valid] < prefix[local_ids[valid] + 1])

    # Check every copied byte in bounded chunks, including packed/scale dtypes.
    # Avoid materializing another full num_rows * topk reference tensor.
    for start in range(0, num_rows, 1024):
        end = min(start + 1024, num_rows)
        mask = valid[start:end].flatten()
        destinations = mapping[start:end].flatten()[mask]
        actual = remapped_hidden_states.view(torch.uint8).index_select(
            0, destinations)
        expected = hidden_states[start:end].view(torch.uint8).repeat_interleave(
            topk, dim=0)[mask]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if scales is not None:
            actual = remapped_scales.view(torch.uint8).index_select(
                0, destinations)
            expected = scales[start:end].view(torch.uint8).repeat_interleave(
                topk, dim=0)[mask]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def ref_init_expert_map(expert_map, local_experts_num, ep_rank, ep_size):
    expert_map_tmp = torch.full((local_experts_num * ep_size, ),
                                -1,
                                dtype=torch.int32,
                                device=DEVICE)

    expert_map_tmp[ep_rank * local_experts_num:ep_rank * local_experts_num +
                   local_experts_num] = torch.arange(local_experts_num,
                                                     device=DEVICE,
                                                     dtype=torch.int32)

    expert_map.copy_(expert_map_tmp)


@pytest.mark.parametrize("local_experts_num", LOCAL_EXPERTS_NUM)
@pytest.mark.parametrize("ep_rank", EP_RANK)
@pytest.mark.parametrize("ep_size", EP_SIZE)
def test_init_expert_map(local_experts_num, ep_rank, ep_size):
    seed_everything(7)

    expert_map = torch.empty((local_experts_num * ep_size),
                             dtype=torch.int32,
                             device=DEVICE)
    ref_expert_map = expert_map.clone()

    ref_init_expert_map(ref_expert_map, local_experts_num, ep_rank, ep_size)
    torch.ops._moe_C.init_expert_map(expert_map, local_experts_num, ep_rank,
                                     ep_size)

    torch.testing.assert_close(expert_map, ref_expert_map, rtol=0, atol=0)


@pytest.mark.parametrize("num_rows", [1, 32, 257])
@pytest.mark.parametrize("hidden_size", [128, 2048])
@pytest.mark.parametrize("total_experts_num", [8, 128])
@pytest.mark.parametrize("topk", [1, 8])
def test_remap_hidden_states_mn_major_padded(num_rows, hidden_size,
                                             total_experts_num, topk):
    """remap_hidden_states can emit the activation scales straight into the
    per-expert MN-major, M-padded-to-4 surface that the mxfp grouped-GEMM
    mainloop indexes, and publishes the per-expert prefix descriptor that lets
    the gemm2 activation quant kernel write the same layout directly."""
    seed_everything(7)

    scale_k = hidden_size // 32
    hidden_states = torch.randint(0,
                                  256, (num_rows, hidden_size),
                                  dtype=torch.uint8,
                                  device=DEVICE).view(
                                      torch.float8_e4m3fn)
    scales = torch.randint(100,
                           140, (num_rows, scale_k),
                           dtype=torch.uint8,
                           device=DEVICE)
    topk_ids = torch.stack([
        torch.randperm(total_experts_num, device=DEVICE)[:topk]
        for _ in range(num_rows)
    ]).to(torch.int32)

    num_moe_inputs = num_rows * topk
    total_padded = num_moe_inputs + 3 * total_experts_num

    def _run(padded):
        rows = total_padded if padded else num_moe_inputs
        out_hs = torch.empty((num_moe_inputs, hidden_size),
                             dtype=hidden_states.dtype,
                             device=DEVICE)
        out_s = torch.zeros((rows, scale_k),
                            dtype=torch.uint8,
                            device=DEVICE)
        rows_per_expert = torch.zeros((total_experts_num),
                                      dtype=torch.int32,
                                      device=DEVICE)
        row_map = torch.empty((num_rows, topk),
                              dtype=torch.int32,
                              device=DEVICE)
        desc = torch.empty((2, total_experts_num + 1),
                           dtype=torch.int32,
                           device=DEVICE) if padded else None
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=scales.view(torch.float8_e8m0fnu),
            remapped_hidden_states=out_hs,
            remapped_hidden_states_scales=out_s.view(torch.float8_e8m0fnu),
            expert_map=None,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=row_map,
            topk_ids=topk_ids,
            total_experts_num=total_experts_num,
            local_experts_num=total_experts_num,
            expert_scale_desc=desc)
        return out_s, rows_per_expert, row_map, desc

    fused, rows_per_expert, row_map, desc = _run(True)

    # The descriptor rows must be the exclusive cumsums of rows and of
    # round_up_4(rows), each with the total appended.
    rpe = rows_per_expert.cpu()
    padded_rpe = (rpe + 3) & ~3
    zero = torch.zeros(1, dtype=torch.int32)
    torch.testing.assert_close(
        desc[0].cpu(), torch.cat([zero,
                                  torch.cumsum(rpe, 0, dtype=torch.int32)]))
    torch.testing.assert_close(
        desc[1].cpu(),
        torch.cat([zero, torch.cumsum(padded_rpe, 0, dtype=torch.int32)]))

    # The row order inside an expert is decided by atomics, so validate the
    # padded surface against the source rows through the emitted row map
    # rather than against a second (independently ordered) kernel run.
    fused_flat = fused.cpu().flatten()
    src = scales.cpu()
    row_map_c = row_map.cpu()
    row_prefix_c = desc[0].cpu()
    scale_prefix_c = desc[1].cpu()
    topk_ids_c = topk_ids.cpu()
    expected = torch.zeros_like(fused_flat)
    for r in range(num_rows):
        for i in range(topk):
            e = int(topk_ids_c[r, i])
            m_local = int(row_map_c[r, i]) - int(row_prefix_c[e])
            base = int(scale_prefix_c[e]) * scale_k
            ld = int(padded_rpe[e])
            idx = base + torch.arange(scale_k) * ld + m_local
            expected[idx] = src[r]
    torch.testing.assert_close(fused_flat, expected, rtol=0, atol=0)

    # ---- Non-padded (expert_scale_desc=None) path + reorder_mxfp_scales ----
    # A second run exercises the desc=None scale output (written row-major and
    # permuted by RemapHiddenStates, with ScaleRemap not launched) and the
    # standalone reorder_mxfp_scales op. Row placement inside an expert is
    # atomic-decided, so this run's order may differ from the padded run above;
    # validate each output against the source rows through *its own* row map
    # instead of comparing bytes across runs.
    plain, rpe2, row_map2, _ = _run(False)
    plain_c = plain.cpu()
    row_map2_c = row_map2.cpu()
    rpe2_c = rpe2.cpu()
    padded_rpe2 = (rpe2_c + 3) & ~3
    row_prefix2 = torch.cat([zero, torch.cumsum(rpe2_c, 0, dtype=torch.int32)])
    scale_prefix2 = torch.cat(
        [zero, torch.cumsum(padded_rpe2, 0, dtype=torch.int32)])

    # desc=None output is row-major permuted: source row r,i lands at row_map2.
    expected_plain = torch.zeros_like(plain_c)
    for r in range(num_rows):
        for i in range(topk):
            expected_plain[int(row_map2_c[r, i])] = src[r]
    torch.testing.assert_close(plain_c, expected_plain, rtol=0, atol=0)

    # reorder_mxfp_scales lifts that permuted surface into the padded MN-major
    # layout; check it against the source rows through the same run's row map.
    ref = torch.ops._moe_C.reorder_mxfp_scales(plain.view(torch.float8_e8m0fnu),
                                               rpe2, total_padded)
    ref_flat = ref.view(torch.uint8).cpu().flatten()
    expected_pad = torch.zeros_like(ref_flat)
    for r in range(num_rows):
        for i in range(topk):
            e = int(topk_ids_c[r, i])
            m_local = int(row_map2_c[r, i]) - int(row_prefix2[e])
            base = int(scale_prefix2[e]) * scale_k
            ld = int(padded_rpe2[e])
            idx = base + torch.arange(scale_k) * ld + m_local
            expected_pad[idx] = src[r]
    torch.testing.assert_close(ref_flat, expected_pad, rtol=0, atol=0)


@pytest.mark.parametrize("hidden_size", [128, 4224])
def test_remap_hidden_states_split_padded(hidden_size):
    # Existing padded cases use <=257 rows, so they miss the split inverse map.
    test_remap_hidden_states_mn_major_padded(513, hidden_size, 32, 8)
