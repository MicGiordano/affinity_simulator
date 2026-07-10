import re

from .affinity_catalog import get_card_spec
from .models import Deck

LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def load_deck_from_text(text: str) -> Deck:
    cards = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith('SIDEBOARD'):
            continue
        m = LINE_RE.match(line)
        if m:
            cards.extend([get_card_spec(m.group(2).strip())] * int(m.group(1)))
    return Deck(cards)
