import { DecimalPipe, PercentPipe } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { Metrics } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';

@Component({
  selector: 'app-reports',
  imports: [DecimalPipe, PercentPipe],
  template: `
    <h2>How well does the digital twin work?</h2>
    <p class="muted">Tested on six months of development data (16 Mar – 15 Sept 2026) where the correct answer
      is already known (simulated with known ground truth). Real field data from Neth Digital is still to come.</p>

    @if (m(); as m) {
      <div class="tiles">
        <div class="card tile">
          <span class="q">How close are the power predictions?</span>
          <span class="value">{{ 100 - m.power.mape_pct | number: '1.0-0' }}% accurate</span>
          <span class="a">For a healthy panel, the predicted power is usually within about
            {{ m.power.mape_pct | number: '1.0-0' }}% of what the panel really produces.</span>
        </div>

        <div class="card tile">
          <span class="q">How well does it follow the real output?</span>
          <span class="value">{{ m.power.r2 | percent: '1.0-0' }}</span>
          <span class="a">of the rises and falls in panel power during the day are explained by the twin.</span>
        </div>

        <div class="card tile">
          <span class="q">How well does it predict panel temperature?</span>
          <span class="value">± {{ m.thermal.rmse_with_wind_c | number: '1.1-1' }} °C</span>
          <span class="a">typical error. Using wind speed makes it
            {{ m.thermal.wind_improvement_pct | number: '1.0-0' }}% more accurate than without wind.</span>
        </div>

        <div class="card tile">
          <span class="q">How reliable is the "normal range"?</span>
          <span class="value">{{ m.conformal.random_split_coverage * 10 | number: '1.0-0' }} out of 10</span>
          <span class="a">healthy readings fall inside the normal range, as designed. When tested on later months
            it was {{ m.conformal.temporal_split_coverage * 10 | number: '1.0-0' }} out of 10.</span>
        </div>

        <div class="card tile">
          <span class="q">Do the AI explanations make sense?</span>
          <span class="value">{{ strength(m.shap_vs_member2.median_row_correlation) }}</span>
          <span class="a">agreement with the AI team's own explanation of why the model chose each fault.</span>
        </div>
      </div>

      <button type="button" class="toggle" (click)="showTech.set(!showTech())" [attr.aria-expanded]="showTech()">
        {{ showTech() ? '▲ Hide technical details' : '▼ Show technical details' }}
      </button>

      @if (showTech()) {
        <div class="tiles">
          <div class="card tile tech"><span class="label">Power RMSE vs truth</span>
            <span class="tvalue">{{ m.power.rmse_mw | number: '1.1-1' }} mW</span>
            <span class="muted">MAPE {{ m.power.mape_pct | number: '1.1-1' }} % · R² {{ m.power.r2 | number: '1.3-3' }}</span></div>
          <div class="card tile tech"><span class="label">Same model, true inputs (ablation)</span>
            <span class="tvalue">{{ m.power.ablation_rmse_mw | number: '1.2-2' }} mW</span>
            <span class="muted">remaining error comes from the light-sensor chain</span></div>
          <div class="card tile tech"><span class="label">Panel temperature RMSE</span>
            <span class="tvalue">{{ m.thermal.rmse_with_wind_c | number: '1.2-2' }} °C</span>
            <span class="muted">{{ m.thermal.rmse_no_wind_c | number: '1.2-2' }} °C without wind ·
              {{ m.thermal.wind_improvement_pct | number: '1.1-1' }} % better</span></div>
          <div class="card tile tech"><span class="label">Faiman thermal fit</span>
            <span class="tvalue">U0 {{ m.thermal.u0 | number: '1.1-1' }} · U1 {{ m.thermal.u1 | number: '1.2-2' }}</span>
            <span class="muted">pvlib defaults 25 · 6.84</span></div>
          <div class="card tile tech"><span class="label">Conformal coverage (target {{ m.conformal.target | percent }})</span>
            <span class="tvalue">{{ m.conformal.random_split_coverage | percent: '1.1-1' }}</span>
            <span class="muted">temporal split {{ m.conformal.temporal_split_coverage | percent: '1.1-1' }}</span></div>
          <div class="card tile tech"><span class="label">SHAP vs Member 2 reference</span>
            <span class="tvalue">r = {{ m.shap_vs_member2.median_row_correlation | number: '1.2-2' }}</span>
            <span class="muted">sign agreement {{ m.shap_vs_member2.sign_agreement | percent: '1.0-0' }} ·
              top feature match {{ m.shap_vs_member2.top1_feature_agreement | percent: '1.0-0' }}</span></div>
        </div>
      }
    }
  `,
  styles: `
    .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin: 16px 0; }
    .tile { display: grid; gap: 6px; align-content: start; }
    .q { font-weight: 600; color: var(--green-700); }
    .value { font-size: 1.8rem; font-weight: 700; color: #1b1b1b; }
    .a { color: #333; line-height: 1.45; }
    .toggle { padding: 8px 14px; border: 1px solid var(--green-700); background: white; color: var(--green-700);
              border-radius: 6px; cursor: pointer; font-weight: 600; }
    .toggle:hover { background: var(--green-50); }
    .tech { background: #fafafa; }
    .label { font-size: .85rem; color: var(--grey-600); }
    .tvalue { font-size: 1.3rem; font-weight: 600; color: #1b1b1b; }
  `,
})
export class Reports {
  private api = inject(TwinApi);
  m = signal<Metrics | null>(null);
  showTech = signal(false);

  constructor() {
    this.api.metrics().subscribe((m) => this.m.set(m));
  }

  strength(r: number): string {
    if (r >= 0.8) return 'Strong';
    if (r >= 0.6) return 'Good';
    if (r >= 0.4) return 'Moderate';
    return 'Weak';
  }
}