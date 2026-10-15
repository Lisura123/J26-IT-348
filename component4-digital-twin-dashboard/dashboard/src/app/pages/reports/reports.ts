import { DecimalPipe, PercentPipe } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { Metrics } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';

/** Evaluation summary of the twin, read from outputs/*.json via GET /metrics. */
@Component({
  selector: 'app-reports',
  imports: [DecimalPipe, PercentPipe],
  template: `
    <h2>Digital-twin evaluation</h2>
    <p class="muted">Evaluation  dataset (16 Mar – 15 Sept 2026). Ground truth is used only for scoring.</p>
    @if (m(); as m) {
      <div class="tiles">
        <div class="card tile"><span class="label">Power RMSE vs truth</span>
          <span class="value">{{ m.power.rmse_mw | number: '1.1-1' }} mW</span>
          <span class="muted">MAPE {{ m.power.mape_pct | number: '1.1-1' }} % · R² {{ m.power.r2 | number: '1.3-3' }}</span></div>
        <div class="card tile"><span class="label">Same model, true inputs</span>
          <span class="value">{{ m.power.ablation_rmse_mw | number: '1.2-2' }} mW</span>
          <span class="muted">remaining error comes from the lux sensor chain</span></div>
        <div class="card tile"><span class="label">Panel temperature RMSE</span>
          <span class="value">{{ m.thermal.rmse_with_wind_c | number: '1.2-2' }} °C</span>
          <span class="muted">{{ m.thermal.rmse_no_wind_c | number: '1.2-2' }} °C without wind ·
            {{ m.thermal.wind_improvement_pct | number: '1.1-1' }} % better</span></div>
        <div class="card tile"><span class="label">Faiman fit</span>
          <span class="value">U0 {{ m.thermal.u0 | number: '1.1-1' }} · U1 {{ m.thermal.u1 | number: '1.2-2' }}</span>
          <span class="muted">pvlib defaults 25 · 6.84</span></div>
        <div class="card tile"><span class="label">Conformal coverage (target {{ m.conformal.target | percent }})</span>
          <span class="value">{{ m.conformal.random_split_coverage | percent: '1.1-1' }}</span>
          <span class="muted">temporal split {{ m.conformal.temporal_split_coverage | percent: '1.1-1' }}</span></div>
        <div class="card tile"><span class="label">SHAP vs Member 2 reference</span>
          <span class="value">r = {{ m.shap_vs_member2.median_row_correlation | number: '1.2-2' }}</span>
          <span class="muted">sign agreement {{ m.shap_vs_member2.sign_agreement | percent: '1.0-0' }}</span></div>
      </div>
    }
  `,
  styles: `
    .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin-top: 16px; }
    .tile { display: grid; gap: 6px; }
    .label { font-size: .85rem; color: var(--grey-600); }
    .value { font-size: 1.7rem; font-weight: 600; color: #1b1b1b; }
  `,
})
export class Reports {
  private api = inject(TwinApi);
  m = signal<Metrics | null>(null);

  constructor() {
    this.api.metrics().subscribe((m) => this.m.set(m));
  }
}