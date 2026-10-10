"""Step 7: reliability-aware client scheduler (history + current link quality + fairness)."""
import numpy as np

class Scheduler:
    def __init__(self, sites, link_stats, theta=0.6, min_sites=3, max_age=3,
                 alpha=0.3, lr=0.2, w_hist=0.5):
        self.sites = list(sites)
        self.stats = link_stats                      # protocol -> (rssi_mean, rssi_std, snr_mean, snr_std)
        self.theta, self.min_sites, self.max_age = theta, min_sites, max_age
        self.alpha, self.lr, self.w_hist = alpha, lr, w_hist
        self.hist = {s: 0.7 for s in self.sites}     # EWMA of on-time delivery (optimistic prior)
        self.age = {s: 0 for s in self.sites}        # rounds since the site last took part
        self.part = {s: 0 for s in self.sites}       # number of rounds taken part in
        # logistic model: P(on-time delivery | link) with features [1, loss, z_rssi, z_snr, online]
        self.w = np.array([1.5, -4.0, 0.4, 0.3, 1.5])

    def features(self, protocol, link):
        rm, rs, sm, ss = self.stats[protocol]
        z_rssi = (float(link["rssi_dbm"]) - rm) / rs
        snr = link["snr_db"]
        z_snr = 0.0 if snr != snr else (float(snr) - sm) / ss      # WiFi has no SNR (NaN)
        return np.array([1.0, float(link["packet_loss_pct"]) / 100.0, z_rssi, z_snr,
                         1.0 if bool(link["online"]) else 0.0])

    def link_prob(self, x):
        return 1.0 / (1.0 + np.exp(-float(self.w @ x)))

    def reliability(self, site, x):
        return self.w_hist * self.hist[site] + (1 - self.w_hist) * self.link_prob(x)

    def select(self, feats):
        """feats: site -> feature vector at round opening. Returns (selected list, reliability dict)."""
        rel = {s: self.reliability(s, feats[s]) for s in self.sites}
        chosen = {s for s in self.sites if rel[s] >= self.theta or self.age[s] >= self.max_age}
        if len(chosen) < self.min_sites:
            rest = sorted((s for s in self.sites if s not in chosen),
                          key=lambda s: rel[s] + 0.05 * self.age[s], reverse=True)
            chosen |= set(rest[: self.min_sites - len(chosen)])
        for s in self.sites:
            if s in chosen:
                self.age[s] = 0; self.part[s] += 1
            else:
                self.age[s] += 1
        return [s for s in self.sites if s in chosen], rel

    def observe(self, site, x, on_time):
        y = 1.0 if on_time else 0.0
        self.hist[site] = (1 - self.alpha) * self.hist[site] + self.alpha * y
        self.w = self.w + self.lr * (y - self.link_prob(x)) * x       # online logistic SGD step

    def jain(self):
        p = np.array([self.part[s] for s in self.sites], float)
        return float(p.sum() ** 2 / (len(p) * (p ** 2).sum())) if p.sum() else 1.0