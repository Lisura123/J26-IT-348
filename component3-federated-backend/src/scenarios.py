"""Step 9: network scenarios. 'trace' replays the shared trace; the others draw events from a seeded generator."""
import numpy as np
import pandas as pd

# Consecutive failed attempts after which a fragment is abandoned (shared trace: failed LoRa uploads show 3 retries)
CAP = {"LORA": 3, "WIFI": 2, "LTE": 3}

SCENARIOS = {
    "trace": None,   # shared files exactly as they are
    "stochastic": dict(extra_outages_per_day=0.0, outage_h=(0.5, 3.0), loss_mult=1.0,
                       sf_min_prob=0.0, deadline_h=5.0),
    "harsh": dict(extra_outages_per_day=4.0, outage_h=(0.5, 3.0), loss_mult=3.0,
                  sf_min_prob=0.8, deadline_h=2.5),
}

def extra_outages(sp, seed, sites, start, hours):
    """Additional random outage windows, the same for every method that uses this seed."""
    if not sp or sp["extra_outages_per_day"] <= 0:
        return pd.DataFrame(columns=["site_id", "start_utc", "end_utc"])
    rng = np.random.default_rng([seed, 11])
    rows = []
    for s in sites:
        for day in range(int(hours // 24) + 1):
            for _ in range(rng.poisson(sp["extra_outages_per_day"])):
                st = start + pd.Timedelta(days=day) + pd.Timedelta(hours=float(rng.uniform(0, 24)))
                dur = pd.Timedelta(hours=float(rng.uniform(*sp["outage_h"])))
                rows.append(dict(site_id=s, start_utc=st, end_utc=st + dur))
    return pd.DataFrame(rows, columns=["site_id", "start_utc", "end_utc"])

def last_online(nr, site, t):
    """Latest link row at or before t that has measurements (offline rows are empty)."""
    rows = nr.lm[(nr.lm.site_id == site) & nr.lm.online]
    before = rows[rows.timestamp_utc <= t]
    return (before if len(before) else rows).iloc[-1] if len(before) else rows.iloc[0]

def draw_context(seed, r, site_idx, protocol, opened, deadline_h, nr, site, sp):
    """Send time, spreading factor and loss probability. Uses its own random stream per (seed, round, site),
    so every method sees the same network luck (common random numbers)."""
    rng = np.random.default_rng([seed, 7, r, site_idx])
    offset = rng.uniform(300, 0.94 * deadline_h * 3600)
    scheduled = opened + pd.Timedelta(seconds=float(round(offset)))
    end = nr.offline_until(site, scheduled)              # an upload waits for the outage to end
    link = last_online(nr, site, end if end is not None else scheduled)
    p = min(float(link["packet_loss_pct"]) * sp["loss_mult"], 90.0) / 100.0
    sf = None
    if protocol == "LORA":
        sf = int(link["lora_sf"])
        if rng.random() < sp["sf_min_prob"]:
            sf = max(sf, 10)
    return dict(rng=rng, scheduled=scheduled, sf=sf, p=p)

def draw_retries(rng, p, frags, protocol):
    """Each fragment is retried until it gets through or CAP failed attempts happen. Returns (retries, failed)."""
    cap, retries = CAP[protocol], 0
    for _ in range(frags):
        fails = 0
        while rng.random() < p:
            fails += 1
            retries += 1
            if fails >= cap:
                return retries, True
    return retries, False

def harsh_link(link, sp, offline):
    """Link as the scheduler sees it at round opening."""
    link = link.copy()
    if link["online"]:
        link["packet_loss_pct"] = min(float(link["packet_loss_pct"]) * sp["loss_mult"], 90.0)
    if offline:
        link["online"] = False
    return link