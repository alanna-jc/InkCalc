def greedy_ctc_decode(logits, input_lengths, blank_idx):
    """
    Greedy CTC decoding for a whole batch.

    logits:        (B, T, C) raw model output (argmax is identical on logits
                   or log-probs, so no softmax needed)
    input_lengths: (B,) true pre-padding lengths — timesteps past this are
                   padding and must not be decoded
    Returns: list of B lists of token indices.

    CTC rules: collapse consecutive repeats first, then remove blanks.
    e.g. [-, 1, 1, -, 1, 2, 2]  ->  [1, 1, 2]   (- is blank)
    The middle blank is what lets CTC output real double tokens like "11".
    """
    best = logits.argmax(dim=-1).cpu()          # (B, T) most likely token per timestep
    decoded = []
    for b in range(best.size(0)):
        seq, prev = [], None
        for t in range(int(input_lengths[b])):
            idx = int(best[b, t])
            if idx != prev and idx != blank_idx:
                seq.append(idx)
            prev = idx
        decoded.append(seq)
    return decoded

def edit_distance(pred: list[int], target: list[int]) -> int:
    """Standard dynamic-programming Levenshtein, O(len(pred) * len(target))."""
    m, n = len(pred), len(target)
    if m == 0: return n
    if n == 0: return m
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            row[j] = min(prev_row[j] + 1,        # delete
                         row[j - 1] + 1,         # insert
                         prev_row[j - 1] + cost) # substitute
        prev_row = row
    return prev_row[n]