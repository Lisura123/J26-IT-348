import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { Alert, CAUSE_LABEL, PRIORITY_LABEL } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

@Component({
  selector: 'app-alerts',
  imports: [DecimalPipe, ColomboTimePipe],
  template: `
    <h2>Alerts</h2>
    <p class="muted">Problems found on the panels, most urgent first. Click a row to see what happened and what to do.</p>
    <div class="toolbar">
      <label>Priority
        <select [value]="priority()" (change)="priority.set($any($event.target).value)">
          <option value="">All</option>
          @for (p of priorities; track p) { <option [value]="p">{{ prioLabel[p] }}</option> }
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
          <tr><th>Priority</th><th>Site</th><th>Problem</th><th>Started</th>
              <th>Power lost</th><th>What to do</th><th>Respond by</th></tr>
        </thead>
        <tbody>
          @for (a of shown(); track a.alert_id) {
            <tr (click)="open(a)" tabindex="0" (keydown.enter)="open(a)">
              <td><span [class]="'prio prio-' + a.priority">{{ icon[a.priority] }} {{ prioLabel[a.priority] }}</span></td>
              <td>{{ a.site_id }}</td>
              <td>{{ causeLabel[a.physics_cause] ?? a.physics_cause }}</td>
              <td>{{ a.start_utc | colomboTime }}</td>
              <td>about {{ a.mean_power_loss_pct | number: '1.0-0' }}%</td>
              <td class="action">{{ a.plain_action }}</td>
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
  causeLabel = CAUSE_LABEL;
  prioLabel = PRIORITY_LABEL;

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