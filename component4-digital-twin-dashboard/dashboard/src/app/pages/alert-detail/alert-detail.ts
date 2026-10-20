import { DecimalPipe, PercentPipe } from '@angular/common';
import { Component, ElementRef, OnDestroy, effect, inject, input, signal, viewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Chart, registerables } from 'chart.js';
import { AlertDetail, CAUSE_LABEL, FEATURE_LABEL, PRIORITY_LABEL, ShapExplanation } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

Chart.register(...registerables);

const UP = '#d84315';
const DOWN = '#1565c0';

@Component({
  selector: 'app-alert-detail',
  imports: [RouterLink, DecimalPipe, PercentPipe, ColomboTimePipe],
  template: `
    <p><a routerLink="/alerts">← All alerts</a></p>
    @if (alert(); as a) {
      <h2>{{ causeLabel[a.physics_cause] ?? a.physics_cause }} · {{ a.site_id }}</h2>
      <p class="muted">Alert {{ a.alert_id }}</p>

      <div class="grid">
        <div class="card">
          <h3>What happened</h3>
          <p class="big">{{ a.plain_summary }}</p>
          <p class="muted">From {{ a.start_utc | colomboTime }} to {{ a.end_utc | colomboTime }} ·
            about {{ a.mean_power_loss_pct | number: '1.0-0' }}% less power than expected</p>
        </div>

        <div class="card action">
          <h3>What to do</h3>
          <p class="big">{{ a.plain_action }}</p>
          <p><strong>{{ prioLabel[a.priority] }}</strong> · for the {{ a.assigned_role.toLowerCase() }} ·
            please respond by {{ a.respond_by_utc | colomboTime }}</p>
        </div>
      </div>

      <button type="button" class="toggle" (click)="showTech.set(!showTech())"
              [attr.aria-expanded]="showTech()">
        {{ showTech() ? '▲ Hide technical details' : '▼ Show technical details' }}
      </button>

      @if (showTech()) {
        <div class="card tech">
          <h3>Technical details</h3>
          <p>{{ a.explanation }}</p>
          <p class="muted">Rule {{ a.rule_id }} · trigger: {{ a.trigger }} · {{ a.n_readings }} readings ·
            respond within {{ a.target_response_h }} h</p>
          <p class="muted">Catalog action: {{ a.recommended_action }}</p>
          <p class="muted">TinyML majority over the episode: {{ a.tinyml_majority }}
            (agrees with physics on {{ a.tinyml_agree_frac | percent: '1.0-0' }} of readings)</p>
        </div>
      }

      @if (a.shap; as s) {
        <div class="card">
          <h3>Why the AI model decided this</h3>
          <p>The AI model on the panel said <strong>{{ causeLabel[s.predicted_fault] ?? s.predicted_fault }}</strong>
            ({{ s.confidence | percent: '1.0-0' }} sure).
            @if (s.predicted_fault !== a.physics_cause) {
              <span class="disagree">▲ This is different from the digital twin's finding, so please double-check.</span>
            }
          </p>
          <div class="chart"><canvas #canvas></canvas></div>
          <p class="muted small">Each bar shows how much one sensor reading influenced the AI's decision.
            <span [style.color]="up">■ pushed towards "{{ causeLabel[s.predicted_fault] ?? s.predicted_fault }}"</span> ·
            <span [style.color]="down">■ pushed away from it</span>. Longer bar = bigger influence.</p>
          @if (showTech()) {
            <p class="muted small">SHAP · reading {{ s.record_id }} · model {{ s.model_version }} ·
              base value {{ s.base_value | number: '1.2-2' }}</p>
          }
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
    .big { font-size: 1.15rem; font-weight: 600; line-height: 1.5; }
    .toggle { margin-bottom: 16px; padding: 8px 14px; border: 1px solid var(--green-700); background: white;
              color: var(--green-700); border-radius: 6px; cursor: pointer; font-weight: 600; }
    .toggle:hover { background: var(--green-50); }
    .tech { margin-bottom: 16px; background: #fafafa; }
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
  showTech = signal(false);
  causeLabel = CAUSE_LABEL;
  prioLabel = PRIORITY_LABEL;
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
    effect(() => {
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
        labels: s.features.map((f) => FEATURE_LABEL[f.feature] ?? f.feature),
        datasets: [{
          label: 'Influence',
          data: s.features.map((f) => f.value),
          backgroundColor: s.features.map((f) => (f.value >= 0 ? UP : DOWN)),
          borderRadius: 4,
          barThickness: 18,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { title: { display: true, text: 'Influence on the AI decision' } },
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