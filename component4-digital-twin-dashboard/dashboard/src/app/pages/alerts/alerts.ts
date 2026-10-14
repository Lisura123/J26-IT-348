import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { Alert } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

/** Alert list (rules R01-R05), most urgent first. Click a row for the explanation. */
@Component({
  selector: 'app-alerts',
  imports: [DecimalPipe, ColomboTimePipe],
  template: `
    <h2>Alerts</h2>
    <div class="toolbar">
      <label>Priority
        <select [value]="priority()" (change)="priority.set($any($event.target).value)">
          <option value="">All</option>
          @for (p of priorities; track p) { <option [value]="p">{{ p }}</option> }
        </select>
      </label>
      <label>Site
        <select [value]="site()" (change)="site.set($any($event.target).value)">
          <option value="">All</option>
          @for (s of sites(); track s) { <option [value]="s">{{ s }}</option> }
        </select>
      </label>
      <span class="muted">{{ shown().length }} of {{ alerts().length }} alerts</span>
    </div>

    <div class="card table-wrap">
      <table>
        <thead>
          <tr><th>Priority</th><th>Site</th><th>Physics cause</th><th>Started</th>
              <th>Readings</th><th>Power loss</th><th>Recommended action</th><th>Respond by</th></tr>
        </thead>
        <tbody>
          @for (a of shown(); track a.alert_id) {
            <tr (click)="open(a)" tabindex="0" (keydown.enter)="open(a)">
              <td><span class="prio" [class]="'prio prio-' + a.priority">{{ icon[a.priority] }} {{ a.priority }}</span></td>
              <td>{{ a.site_id }}</td>
              <td>{{ a.physics_cause }}</td>
              <td>{{ a.start_utc | colomboTime }}</td>
              <td>{{ a.n_readings }}</td>
              <td>{{ a.mean_power_loss_pct | number: '1.0-0' }} %</td>
              <td class="action">{{ a.recommended_action }}</td>
              <td>{{ a.respond_by_utc | colomboTime }}</td>
            </tr>
          }
        </tbody>
      </table>
    </div>
  `,
  styles: `
    .toolbar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 12px; }
    select { margin-left: 8px; padding: 6px; }
    .table-wrap { overflow-x: auto; padding: 0; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: top; }
    th { background: var(--green-50); position: sticky; top: 0; }
    tbody tr { cursor: pointer; }
    tbody tr:hover, tbody tr:focus { background: #fafafa; outline: none; }
    .action { max-width: 320px; }
    .prio { font-weight: 600; white-space: nowrap; }
    .prio-CRITICAL { color: var(--red-600); }
    .prio-HIGH { color: #e65100; }
    .prio-MEDIUM { color: #8d6e00; }
  `,
})
export class Alerts {
  private api = inject(TwinApi);
  private router = inject(Router);

  priorities = ['CRITICAL', 'HIGH', 'MEDIUM'];
  icon: Record<string, string> = { CRITICAL: '■', HIGH: '◆', MEDIUM: '▲', LOW: '●', INFO: '○' };

  alerts = signal<Alert[]>([]);
  priority = signal('');
  site = signal('');
  sites = computed(() => [...new Set(this.alerts().map((a) => a.site_id))].sort());
  shown = computed(() =>
    this.alerts().filter((a) => (!this.priority() || a.priority === this.priority()) &&
                                (!this.site() || a.site_id === this.site())));

  constructor() {
    this.api.alerts().subscribe((a) => this.alerts.set(a));
  }

  open(a: Alert): void {
    this.router.navigate(['/alerts', a.alert_id]);
  }
}