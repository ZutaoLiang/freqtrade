"""W67tJltC with the author's ATR stop switched back on - zero-cross entries.

This is :class:`ZEdgeSignal` plus one single change: the Pine ``use_stop_loss``
module (a fixed stop at ``close -/+ ATR(14) * 2``, measured on the signal bar
and never moved). Position sizing stays at the matrix's fixed notional stake,
so this unit is a clean A/B against the ``ZEdgeSignal`` matrix: same signals,
same sizing, stop on versus stop off.

It is therefore *not* the author's full configuration either - the Pine risk
sizing (1% of a static equity input divided by the stop distance) is still
replaced by the fixed stake. ``ZEdgeFull`` carries both.

``stop_loss`` exits are a real strategy exit here, not the -99% engine floor;
the two are told apart by the exit rate relative to the ATR distance.
"""

from __future__ import annotations

from zedge_core import AtrStopMixin
from ZEdgeSignal import ZEdgeSignal


class ZEdgeStopSignal(AtrStopMixin, ZEdgeSignal):
    use_custom_stoploss = True
    use_stop_loss = True
