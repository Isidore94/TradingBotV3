"""Desk Link: LAN relay between the main desk and view-only satellites.

Tier 1 of docs/MULTI_MACHINE_DESK_PROPOSAL.md. The main PC stays the sole
engine (TWS, scanners, shared-state writer); satellites connect over the
local network to mirror state and receive live alert chart popups. This
package is transport + protocol only and stays PySide6-free so it can be
unit-tested headless; the GUI wires it up through thin services.
"""

from desk_link.protocol import (  # noqa: F401
    PROTOCOL_VERSION,
    DeskLinkAuthError,
    DeskLinkProtocolError,
    decode_message,
    encode_message,
    make_hello,
    make_message,
    make_rejected,
    make_welcome,
)
from desk_link.server import DeskLinkServer  # noqa: F401
from desk_link.client import DeskLinkClient  # noqa: F401
