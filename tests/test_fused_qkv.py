import pytest
import torch

from inferengine.kernels.fused_qkv import fused_qkv, pack_qkv_weights


def test_pack_qkv_weights_matches_linear_layout():
    q = torch.randn(12, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    packed = pack_qkv_weights(q, k, v)
    assert packed.shape == (8, 20)
    x = torch.randn(3, 8)
    expected = torch.cat((x @ q.T, x @ k.T, x @ v.T), dim=-1)
    torch.testing.assert_close(x @ packed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_fused_qkv_matches_torch_cuda():
    pytest.importorskip("triton")
    x = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    q = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    packed = pack_qkv_weights(q, k, v)
    actual = torch.cat(fused_qkv(x, packed, (128, 32, 32)), dim=-1)
    expected = x @ packed
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
