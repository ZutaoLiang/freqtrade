"""W67tJltC with the ATR stop on and ``Threshold Reversion`` entries.

:class:`ZEdgeReversion` plus the Pine ``use_stop_loss`` module; see
``ZEdgeStopSignal`` for what the A/B isolates.
"""

from __future__ import annotations

from zedge_core import AtrStopMixin
from ZEdgeReversion import ZEdgeReversion


class ZEdgeStopReversion(AtrStopMixin, ZEdgeReversion):
    use_custom_stoploss = True
    use_stop_loss = True
