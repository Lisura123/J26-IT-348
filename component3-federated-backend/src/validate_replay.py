import pandas as pd
from src import data_loader as dl
from src.network_replay import NetworkReplay

tx = dl.load_transmissions()
nr = NetworkReplay()
rows = []
for r in tx.itertuples():
    sf = int(r.lora_sf) if r.protocol == "LORA" else None
    o = nr.replay(r.site_id, r.protocol, r.scheduled_send_utc, r.deadline_utc,
                  r.payload_bytes, sf, r.retries)
    rows.append(dict(protocol=r.protocol, **o, f_tx_start=r.tx_start_utc,
                     f_wait=r.waited_offline_s, f_frag=r.fragments,
                     f_air=r.airtime_s, f_dur=r.transfer_duration_s,
                     f_status=r.status))
d = pd.DataFrame(rows)

print("tx_start exact       :", round((d.tx_start_utc == d.f_tx_start).mean() * 100, 1), "%")
print("waited_offline exact :", round(((d.waited_offline_s - d.f_wait).abs() < 1).mean() * 100, 1), "%")
print("LATE status matches  :", round((d.late == (d.f_status == "LATE")).mean() * 100, 1), "%")

lo = d[d.protocol == "LORA"]
print("LoRa fragments exact :", round((lo.fragments == lo.f_frag).mean() * 100, 1), "%")
ok = lo[lo.f_status != "FAILED"]          # failed uploads stop early, so exclude them
print("LoRa airtime error % :", round(((ok.airtime_s - ok.f_air).abs() / ok.f_air).max() * 100, 2), "(max, non-failed)")
print("LoRa duration error %:", round(((ok.transfer_duration_s - ok.f_dur).abs() / ok.f_dur).max() * 100, 2), "(max, non-failed)")