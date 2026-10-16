import math
from src import data_loader as dl

DUTY_CYCLE = 0.01                  # LoRa 1 % -> wait = airtime * 99
MAX_FRAME = {7: 222, 8: 222, 9: 115, 10: 51, 11: 51, 12: 51}   # payload bytes per frame

# Semtech formula underestimates the shared trace; these factors calibrate it
# (observed airtime / formula airtime for a 297 byte payload).
# SF11 and SF12 do not appear in the trace, so they reuse the SF10 factor (assumption).
CAL = {7: 1.783, 8: 1.746, 9: 1.329, 10: 1.083, 11: 1.083, 12: 1.083}

def lora_fragments(payload_bytes, sf):
    return math.ceil(payload_bytes / MAX_FRAME[int(sf)])

def lora_frame_airtime(frame_bytes, sf, bw=125000, cr=1, preamble=8, crc=1, ih=0):
    """Semtech time on air for one frame, in seconds."""
    sf = int(sf)
    ts = 2 ** sf / bw
    de = 1 if ts > 0.016 else 0
    n = 8 + max(math.ceil((8 * frame_bytes - 4 * sf + 28 + 16 * crc - 20 * ih)
                          / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25 + n) * ts

def lora_airtime(payload_bytes, sf, retries=0):
    """All fragments once, plus each retry re-sends one average fragment."""
    sf = int(sf)
    frames, left = [], payload_bytes
    while left > 0:
        frames.append(min(MAX_FRAME[sf], left))
        left -= frames[-1]
    base = sum(lora_frame_airtime(f, sf) for f in frames)
    return CAL[sf] * (base + retries * base / len(frames))

class NetworkReplay:
    def __init__(self):
        self.out = dl.load_outages()
        self.lm = dl.load_link_metrics().sort_values("timestamp_utc")

    def offline_until(self, site_id, t):
        o = self.out[(self.out.site_id == site_id)
                     & (self.out.start_utc <= t) & (self.out.end_utc > t)]
        return o.end_utc.max() if len(o) else None

    def link_at(self, site_id, t):
        """Latest 10 minute link row at or before t (gives rssi, snr, lora_sf, loss)."""
        s = self.lm[(self.lm.site_id == site_id) & (self.lm.timestamp_utc <= t)]
        return s.iloc[-1] if len(s) else None

    def replay(self, site_id, protocol, scheduled, deadline,
               payload_bytes, sf=None, retries=0):
        end = self.offline_until(site_id, scheduled)
        tx_start = end if end is not None else scheduled
        waited = (tx_start - scheduled).total_seconds()
        if protocol == "LORA":
            frags = lora_fragments(payload_bytes, sf)
            air = lora_airtime(payload_bytes, sf, retries)
            wait = air * (1 / DUTY_CYCLE - 1)
            transfer = air + wait
        else:
            frags, air, wait, transfer = 1, None, 0.0, None
        return dict(tx_start_utc=tx_start, waited_offline_s=waited,
                    fragments=frags, airtime_s=air, duty_cycle_wait_s=wait,
                    transfer_duration_s=transfer, late=tx_start > deadline)