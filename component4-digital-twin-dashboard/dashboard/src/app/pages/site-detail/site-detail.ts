import { DecimalPipe } from '@angular/common';
import { Component, ElementRef, OnDestroy, computed, effect, inject, input, signal, viewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Chart, registerables } from 'chart.js';
import { TimePoint } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { toColombo } from '../../shared/colombo-time';

Chart.register(...registerables);

const DAY = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Colombo' });   // YYYY-MM-DD in Colombo

/**
 * Panel detail: actual power vs the twin's expected power, inside the 90 % conformal band.
 * Readings outside the band are drawn as separate red points (shape + legend, not colour alone).
 */
@Component({
  selector: 'app-site-detail',
  imports: [RouterLink, DecimalPipe],
  template: `
    <p><a routerLink="/">← All sites</a></p>
    <h2>{{ id() }} – actual vs digital-twin expected power</h2>

    <div class="toolbar">
      <label>Day (Colombo time)
        <select [value]="day()" (change)="day.set($any($event.target).value)">
          @for (d of days(); track d) { <option [value]="d">{{ d }}</option> }
        </select>
      </label>
      <span class="muted">{{ shown().length }} readings ·
        {{ outside() }} outside the 90 % band ({{ outsidePct() | number: '1.0-0' }} %)</span>
    </div>

    <div class="card chart"><canvas #canvas></canvas></div>

    <div class="card">
      <h3>Physics cause for this day</h3>
      <table>
        <tr><th>Cause</th><th>Readings</th></tr>
        @for (c of causes(); track c[0]) { <tr><td>{{ c[0] }}</td><td>{{ c[1] }}</td></tr> }
      </table>
    </div>
  `,
  styles: `
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-bottom: 12px; }
    select { margin-left: 8px; padding: 6px; }
    .chart { height: 420px; margin-bottom: 16px; }
    table { border-collapse: collapse; }
    th, td { text-align: left; padding: 4px 16px 4px 0; border-bottom: 1px solid #eee; }
  `,
})
export class SiteDetail implements OnDestroy {
  private api = inject(TwinApi);

  id = input.required<string>();                 // from the route  /sites/:id
  canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');

  points = signal<TimePoint[]>([]);
  day = signal('');
  days = computed(() => [...new Set(this.points().map((p) => DAY.format(new Date(p.timestamp_utc))))]);
  shown = computed(() => this.points().filter((p) => DAY.format(new Date(p.timestamp_utc)) === this.day()));
  outside = computed(() => this.shown().filter((p) => !p.within_expected_band).length);
  outsidePct = computed(() => (this.shown().length ? (100 * this.outside()) / this.shown().length : 0));
  causes = computed(() => {
    const n = new Map<string, number>();
    this.shown().forEach((p) => n.set(p.physics_cause, (n.get(p.physics_cause) ?? 0) + 1));
    return [...n.entries()].sort((a, b) => b[1] - a[1]);
  });

  private chart?: Chart;

  constructor() {
    effect(() => {                               // reload when the route id changes
      this.api.timeseries(this.id()).subscribe((pts) => {
        this.points.set(pts);
        this.day.set(this.days()[0] ?? '');
      });
    });
    effect(() => this.draw(this.shown()));       // redraw when the day changes
  }

  private draw(pts: TimePoint[]): void {
    this.chart?.destroy();
    if (!pts.length) return;
    this.chart = new Chart(this.canvas().nativeElement, {
      type: 'line',
      data: {
        labels: pts.map((p) => toColombo(p.timestamp_utc)),
        datasets: [
          { label: '90 % band (low)', data: pts.map((p) => p.power_band_low_mw),
            borderWidth: 0, pointRadius: 0, fill: false },
          { label: '90 % conformal band', data: pts.map((p) => p.power_band_high_mw),
            borderWidth: 0, pointRadius: 0, fill: '-1', backgroundColor: 'rgba(46,125,50,0.15)' },
          { label: 'Twin expected', data: pts.map((p) => p.expected_power_mw),
            borderColor: '#455a64', backgroundColor: '#455a64', borderDash: [6, 4], borderWidth: 2, pointRadius: 0 },
          { label: 'Actual', data: pts.map((p) => p.power_mw),
            borderColor: '#2e7d32', backgroundColor: '#2e7d32', borderWidth: 2, pointRadius: 0 },
          { label: 'Outside band', data: pts.map((p) => (p.within_expected_band ? null : p.power_mw)),
            showLine: false, pointRadius: 4, pointStyle: 'rectRot',
            pointBackgroundColor: '#c62828', backgroundColor: '#c62828', borderColor: '#fff', borderWidth: 1 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },     // hover shows every series at that time
        scales: {
          x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } },
          y: { title: { display: true, text: 'Power (mW)' }, beginAtZero: true },
        },
        plugins: {
          legend: { labels: { filter: (item) => item.text !== '90 % band (low)' } },
          tooltip: { filter: (item) => item.dataset.label !== '90 % band (low)' },
        },
      },
    });
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }
}