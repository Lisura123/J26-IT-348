import { DecimalPipe } from '@angular/common';
import { Component, effect, inject, signal } from '@angular/core';
import { DeviceHealth } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

const VALUE_LABEL: Record<string, [string, string]> = {
  voltage_v: ['Voltage', 'V'],
  current_ma: ['Current', 'mA'],
  power_mw: ['Power', 'mW'],
  irradiance_lux: ['Light', 'lux'],
  panel_temp1_c: ['Panel temperature', '°C'],
  ambient_temp_c: ['Air temperature', '°C'],
  dust_lpo_pct: ['Dust (LPO)', '%'],
  health_score: ['Health score', '/100'],
};

const STATUS_LABEL: Record<string, string> = {
  HEALTHY: '● Healthy',
  DEGRADED: '▲ Degraded',
  FAULTY: '■ Faulty',
  NOT_GRADED: '○ Not graded',
  UNKNOWN: '○ No status sent',
  MISSING: '○ No status sent',
};

const SEVERITY_ICON: Record<string, string> = { CRITICAL: '■', WARNING: '▲', INFO: '●', OK: '✓' };

@Component({
  selector: 'app-device-health',
  imports: [DecimalPipe, ColomboTimePipe],
  template: `
    <h2>Device health</h2>
    <p class="muted">What the sensors on each device are reporting, what it means, and what to check.</p>

    <div class="toolbar">
      <label>Site
        <select [value]="site()" (change)="site.set($any($event.target).value)">
          @for (s of sites(); track s) { <option [value]="s">{{ s }}</option> }
        </select>
      </label>
      @if (data(); as d) {
        <span class="muted">{{ d.device_id }} · last {{ d.n_readings }} readings ·
          {{ d.from_utc | colomboTime }} – {{ d.to_utc | colomboTime }} · source: {{ d.collection }}</span>
      }
    </div>

    @if (loading()) { <p class="muted">Loading…</p> }
    @if (error()) { <p class="error">{{ error() }}</p> }

    @if (data(); as d) {
      <div class="card latest">
        <div class="head">
          <h3>Latest reading</h3>
          <span [class]="'status st-' + d.latest.health_status">{{ statusLabel[d.latest.health_status] ?? d.latest.health_status }}</span>
        </div>
        <p class="big">{{ d.latest.status_text }}</p>
        @if (d.latest.context) { <p class="note">ℹ {{ d.latest.context }}</p> }
        <p class="muted">{{ d.latest.timestamp_utc | colomboTime }} · record {{ d.latest.record_id }}</p>

        <div class="values">
          @for (k of keys(d.latest.values); track k) {
            <div class="value">
              <span class="label">{{ valueLabel[k]?.[0] ?? k }}</span>
              <span class="num">{{ d.latest.values[k] | number: '1.0-2' }} <small>{{ valueLabel[k]?.[1] }}</small></span>
            </div>
          }
        </div>

        @for (i of d.latest.issues; track i.code) {
          <p [class]="'sev sev-' + i.severity">{{ sevIcon[i.severity] }} {{ i.sensor }}: {{ i.what }}</p>
        }
      </div>

      <h3>Issues in the last {{ d.n_readings }} readings</h3>
      <div class="chips">
        @for (k of keys(d.status_counts); track k) {
          <span class="chip">{{ statusLabel[k] ?? k }}: {{ d.status_counts[k] }}</span>
        }
      </div>

      @if (d.issues.length === 0) {
        <div class="card ok">✓ No sensor issues were reported in these readings.</div>
      }
      @for (i of d.issues; track i.code) {
        <div [class]="'card issue issue-' + i.severity">
          <div class="head">
            <strong [class]="'sev-' + i.severity">{{ sevIcon[i.severity] }} {{ i.severity }} · {{ i.sensor }}</strong>
            <span class="muted">{{ i.count }} × · last {{ i.last_seen_utc | colomboTime }}</span>
          </div>
          <p class="code">{{ i.code }}</p>
          <dl>
            <dt>What happened</dt><dd>{{ i.what }}</dd>
            <dt>What to check</dt><dd>{{ i.check }}</dd>
            <dt>Effect on the digital twin</dt><dd>{{ i.twin }}</dd>
          </dl>
        </div>
      }
    }
  `,
  styles: `
    .toolbar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 16px; }
    select { margin-left: 8px; padding: 6px; }
    .card { margin-bottom: 16px; }
    .head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; }
    .head h3 { margin: 0; }
    .big { font-size: 1.1rem; font-weight: 600; margin: 8px 0; }
    .note { background: #e3f2fd; padding: 8px 12px; border-radius: 6px; margin: 8px 0; }
    .values { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin: 12px 0; }
    .value { display: grid; background: var(--green-50); padding: 8px 10px; border-radius: 6px; }
    .label { font-size: .8rem; color: #555; }
    .num { font-size: 1.2rem; font-weight: 600; }
    .status { font-weight: 700; padding: 4px 10px; border-radius: 12px; background: #eee; }
    .st-HEALTHY { color: var(--green-700); }
    .st-DEGRADED { color: #8d6e00; }
    .st-FAULTY { color: var(--red-600); }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .chip { background: #eee; padding: 4px 10px; border-radius: 12px; font-size: .85rem; }
    .issue { border-left: 6px solid #90a4ae; }
    .issue-CRITICAL { border-left-color: var(--red-600); }
    .issue-WARNING { border-left-color: #f9a825; }
    .issue-INFO { border-left-color: #1565c0; }
    .sev-CRITICAL { color: var(--red-600); }
    .sev-WARNING { color: #8d6e00; }
    .sev-INFO { color: #1565c0; }
    .sev { margin: 4px 0; font-weight: 600; }
    .code { font-family: monospace; font-size: .8rem; color: #666; margin: 4px 0 8px; }
    dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; }
    dt { font-weight: 600; color: #444; }
    dd { margin: 0; }
    .ok { color: var(--green-700); font-weight: 600; }
    .error { color: var(--red-600); }
  `,
})
export class DeviceHealthPage {
  private api = inject(TwinApi);

  valueLabel = VALUE_LABEL;
  statusLabel = STATUS_LABEL;
  sevIcon = SEVERITY_ICON;

  sites = signal<string[]>(['SITE-A']);
  site = signal('SITE-A');
  data = signal<DeviceHealth | null>(null);
  loading = signal(false);
  error = signal('');

  constructor() {
    this.api.sites().subscribe((s) => this.sites.set(s.map((x) => x.site_id)));
    effect(() => this.load(this.site()));
  }

  keys(o: object): string[] {
    return Object.keys(o);
  }

  private load(siteId: string): void {
    this.data.set(null);
    this.error.set('');
    this.loading.set(true);
    this.api.deviceHealth(siteId).subscribe({
      next: (d) => { this.data.set(d); this.loading.set(false); },
      error: (e) => {
        this.loading.set(false);
        this.error.set(e.status === 404 ? `No device data for ${siteId} yet.`
          : e.status === 503 ? 'The device database cannot be reached right now. Please try again.'
          : `Could not load device health (HTTP ${e.status}).`);
      },
    });
  }
}