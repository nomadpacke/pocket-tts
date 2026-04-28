# Quantization

The legacy PyTorch implementation of pocket-tts shipped a dynamic int8 path
backed by `torchao` / `torch.ao`. The MAX rewrite has not yet ported that
path, so `quantize=True` raises `NotImplementedError`.

For now, please run with `quantize=False` (the default).

If you'd like int8 support, please follow the
[issue tracker](https://github.com/kyutai-labs/pocket-tts/issues).
