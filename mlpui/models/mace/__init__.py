"""Bundled MACE 0.3.16; see LICENSE.md and ../VENDORED_MODELS.md."""
from contextlib import nullcontext
import torch

# e3nn 0.4.4's installed constants.pt contains built-in slices. Allow only
# this harmless type during that import, without disabling weights-only loading
# or changing the trust policy for any user checkpoint.
with (torch.serialization.safe_globals([slice])
      if hasattr(torch.serialization, "safe_globals") else nullcontext()):
    from e3nn import o3
