"""Second pure-signal variant of W67tJltC: the ``Threshold Reversion`` mode.

Identical to :class:`ZEdgeSignal` except for the Pine ``entry_mode`` input:
entries fire when the smoothed composite crosses back up through -1.5 (long) or
back down through +1.5 (short), while the exits stay on the default zero level.

That makes this variant structurally different from the zero-cross default: the
exit condition is no longer the same expression as the opposite entry, so the
strategy is genuinely flat between signals instead of always in the market.
Everything else - the removed ATR stop, the removed risk sizing, the fixed
stake and the engine disclosures - is unchanged; see ZEdgeSignal's docstring.
"""

from __future__ import annotations

from ZEdgeSignal import ZEdgeSignal


class ZEdgeReversion(ZEdgeSignal):
    entry_mode = "Threshold Reversion"
