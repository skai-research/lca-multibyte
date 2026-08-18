import torch


def downsample_mean(boundaries, hidden):
    """Downsample by taking the mean of each boundary-defined group.

    Minimal version: no null-group handling, no shifting, no extra conventions.

    Semantics: a value of 1 at position t means token t is the *last* token in the group
    (i.e. the split happens *after* t).

    Args:
        boundaries: [B, L] with 0/1 boundary indicators.
        hidden: [L, B, D]

    Returns:
        shortened_hidden: [S, B, D] where S depends on the max number of groups in the batch.
    """
    boundaries = (boundaries != 0).to(dtype=torch.long)

    B, L = boundaries.shape
    if L == 0:
        return hidden.new_zeros((0, B, hidden.size(-1)))

    # [L, B, D] -> [B, L, D]
    x = hidden.transpose(0, 1)
    D = x.size(-1)

    # Group id per token; includes the trailing "last" group.
    # With the semantics above, boundary tokens belong to the current group, and the next token
    # begins the next group.
    group_id = boundaries.cumsum(dim=1) - boundaries  # [B, L], int64
    S = int(group_id.max().item()) + 1

    sums = x.new_zeros((B, S, D))
    sums.scatter_add_(dim=1, index=group_id.unsqueeze(-1).expand(-1, -1, D), src=x)

    counts = x.new_zeros((B, S))
    counts.scatter_add_(dim=1, index=group_id, src=torch.ones_like(group_id, dtype=counts.dtype))
    counts = counts.clamp_min(1.0).unsqueeze(-1)

    means = sums / counts  # [B, S, D]
    return means.transpose(0, 1).contiguous()


def final(foo,
          upsample):
    """
        Input:
            B x L x S
    """
    autoregressive = foo != 0
    lel = 1 - foo

    lel[autoregressive] = 0

    dim = 2 if upsample else 1

    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

    return lel

@torch._dynamo.disable
def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()

    if upsample:
        n_segments += 1

    if n_segments == 0:
        return None

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device,
        dtype=boundaries.dtype,
    )

    hh1 = boundaries.cumsum(1) 

    if not upsample:
        hh1 -= boundaries #  i.e a tesnor that counts from 0 to n_segments is reduce by either 1 or 0

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample(boundaries, hidden, null_group, use_group_attn=False):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 ends a new group
        - We append an extra "null" group at the beginning
        - We discard last group because it won't be used (in terms of upsampling)

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D
    """

    foo = common(boundaries, upsample=False)  # B x L x S

    if foo is None:
        return null_group.repeat(1, hidden.size(1), 1)
    else:
        bar = final(foo=foo, upsample=False)  # B x L x S

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
        shortened_hidden = torch.cat(
            [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
        )
        return shortened_hidden


def upsample(boundaries, shortened_hidden):
    """
        Upsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - i-th group can be upsampled only to the tokens from (i+1)-th group, otherwise there's a leak

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D
    """

    foo = common(boundaries, upsample=True)  # B x L x S
    bar = final(foo, upsample=True)  # B x L x S

    return torch.einsum('sbd,bls->lbd', shortened_hidden, bar)
