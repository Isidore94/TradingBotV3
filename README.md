# TradingBotV3

Scripts:
- `scripts/master_avwap.py` – daily AVWAP + previous-AVWAP engine.
- `scripts/bounce_bot.py` – intraday 5-minute bounce detector.

Inputs:
- `longs.txt`
- `shorts.txt`

Outputs:
- `output/master_avwap_events.txt`
- `output/bouncers.txt`

Basic usage:

```bash
pip install -r requirements.txt

python scripts/master_avwap.py   # run daily AVWAP engine
python scripts/bounce_bot.py --use_gui  # run intraday bounce bot
