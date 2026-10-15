import { DecimalPipe, PercentPipe } from '@angular/common';
import { Component, ElementRef, OnDestroy, effect, inject, input, signal, viewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Chart, registerables } from 'chart.js';
import { AlertDetail, ShapExplanation } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

Chart.register(...registerables);

const UP = '#d84315';     // pushes the model TOWARDS its predicted fault
const DOWN = '#1565c0';   // pushes the model AWAY from it


@Component({
  selector: 'app-alert-detail',
  imports: [RouterLink, DecimalPipe, PercentPipe, ColomboTimePipe],
  template: `
    <p><a routerLink="/alerts">← All alerts</a></p>
    @if (alert(); as a) {
      <h2>{{ a.alert_id }} · {{ a.site_id }} · {{ a.physics_cause }}</h2>

      <div class="grid">
        <div class="card">
          <h3>1 · What the digital twin found</h3>
          <p>{{ a.explanation }}</p>
          <p class="muted">{{ a.n_readings }} readings from {{ a.start_utc | colomboTime }} to
            {{ a.end_utc | colomboTime }} · mean power loss {{ a.mean_power_loss_pct | number: '1.0-0' }} %</p>
        </div>

        <div class="card action">
          <h3>2 · Recommended action ({{ a.rule_id }})</h3>
          <p class="big">{{ a.recommended_action }}</p>
          <p><strong>{{ a.priority }}</strong> · assigned to {{ a.assigned_role }} ·
            respond within {{ a.target_response_h }} h (by {{ a.respond_by_utc | colomboTime }})</p>
          <p class="muted">Rule trigger: {{ a.trigger }}</p>
          <p class="muted">TinyML majority over the episode: {{ a.tinyml_majority }}
            (agrees with physics on {{ a.tinyml_agree_frac | percent: '1.0-0' }} of readings)</p>
        </div>
      </div>

      @if (a.shap; as s) {
        <div class="card">
          <h3>3 · Why TinyML predicted what it did (SHAP)</h3>
          <p>Reading <code>{{ s.record_id }}</code> · model <strong>{{ s.model_version }}</strong> predicted
            <strong>{{ s.predicted_fault }}</strong> ({{ s.confidence | percent: '1.0-0' }}) ·
            base value {{ s.base_value | number: '1.2-2' }}
            @if (s.predicted_fault !== a.physics_cause) {
              <span class="disagree">▲ disagrees with physics ({{ a.physics_cause }}) — review</span>
            }
          </p>
          <div class="chart"><canvas #canvas></canvas></div>
          <p class="muted small">Bars show each sensor's contribution to the predicted-fault probability.
            <span [style.color]="up">■ pushes towards {{ s.predicted_fault }}</span> ·
            <span [style.color]="down">■ pushes away</span>.
            Base value + all bars = {{ s.confidence | percent: '1.0-0' }}.</p>
        </div>
      }
    } @else if (error()) {
      <p class="error">{{ error() }}</p>
    }
  `,
  styles: `
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 16px; }
    .card p { margin: 6px 0; }
    .action { border-left: 6px solid var(--green-700); }
    .big { font-size: 1.1rem; font-weight: 600; }
    .chart { height: 300px; }
    .small { font-size: .8rem; }
    .disagree { color: #8d6e00; font-weight: 600; margin-left: 8px; }
    .error { color: var(--red-600); }
  `,
})
export class AlertDetailPage implements OnDestroy {
  private api = inject(TwinApi);

  id = input.required<string>();
  canvas = viewChild<ElementRef<HTMLCanvasElement>>('canvas');
  alert = signal<AlertDetail | null>(null);
  error = signal('');
  up = UP;
  down = DOWN;
  private chart?: Chart;

  constructor() {
    effect(() => {
      this.api.alert(this.id()).subscribe({
        next: (a) => this.alert.set(a),
        error: (e) => this.error.set(`Could not load alert (HTTP ${e.status})`),
      });
    });
    effect(() => {                                   // draw once the canvas exists
      const el = this.canvas()?.nativeElement;
      const shap = this.alert()?.shap;
      if (el && shap) this.draw(el, shap);
    });
  }

  private draw(el: HTMLCanvasElement, s: ShapExplanation): void {
    this.chart?.destroy();
    this.chart = new Chart(el, {
      type: 'bar',
      data: {
        labels: s.features.map((f) => f.feature),
        datasets: [{
          label: 'SHAP value',
          data: s.features.map((f) => f.value),
          backgroundColor: s.features.map((f) => (f.value >= 0 ? UP : DOWN)),
          borderRadius: 4,
          barThickness: 18,
        }],
      },
      options: {
        indexAxis: 'y',                              // horizontal bars, largest effect on top
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { title: { display: true, text: 'Contribution to predicted-fault probability' } },
          y: { grid: { display: false } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${(c.raw as number) >= 0 ? '+' : ''}${(c.raw as number).toFixed(3)}` } },
        },
      },
    });
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }
}