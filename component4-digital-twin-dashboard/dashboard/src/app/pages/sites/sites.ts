import { DecimalPipe } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Site } from '../../core/models';
import { TwinApi } from '../../core/twin-api.service';
import { ColomboTimePipe } from '../../shared/colombo-time';

/** Site overview: one card per panel, coloured by its worst open alert. */
@Component({
  selector: 'app-sites',
  imports: [RouterLink, DecimalPipe, ColomboTimePipe],
  template: `
    <h2>Site overview</h2>
    @if (error()) { <p class="error">{{ error() }}</p> }
    <div class="grid">
      @for (s of sites(); track s.site_id) {
        <a class="card site" [routerLink]="['/sites', s.site_id]" [class]="'site status-' + s.status">
          <div class="head">
            <strong>{{ s.display_name }}</strong>
            <span class="status">{{ icon[s.status] }} {{ s.status }}</span>
          </div>
          <p class="muted">{{ s.panel_id }} · {{ s.protocol }}</p>

          @if (s.latest; as l) {
            <p class="power">
              <span class="big">{{ l.power_mw | number: '1.0-0' }}</span> mW
              <span class="muted">/ expected {{ l.expected_power_mw | number: '1.0-0' }}</span>
            </p>
            <p>Latest: <strong>{{ l.physics_cause }}</strong>
              <span class="muted"> · {{ l.timestamp_utc | colomboTime }}</span></p>
          }
          <p>{{ s.open_alerts }} alert(s){{ s.worst_priority ? ' · worst ' + s.worst_priority : '' }}</p>

          @if (s.model; as m) {
            <p class="muted small">
              TinyML {{ m.model_version }}{{ m.model_up_to_date ? '' : ' (latest ' + m.latest_published_version + ')' }}
              · link {{ m.link_quality }}
            </p>
          }
        </a>
      }
    </div>
  `,
  styles: `
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .site { display: grid; gap: 6px; text-decoration: none; color: inherit; border-left: 6px solid #9e9e9e; }
    .site p { margin: 0; }
    .site:hover { box-shadow: 0 3px 10px rgba(0,0,0,.15); }
    .head { display: flex; justify-content: space-between; gap: 8px; }
    .status { font-size: .8rem; font-weight: 600; }
    .status-OK { border-left-color: var(--green-700); }
    .status-WARNING { border-left-color: var(--amber-600); }
    .status-FAULT { border-left-color: var(--red-600); }
    .big { font-size: 1.6rem; font-weight: 600; }
    .small { font-size: .8rem; }
    .error { color: var(--red-600); }
  `,
})
export class Sites {
  private api = inject(TwinApi);
  sites = signal<Site[]>([]);
  error = signal('');
  icon: Record<string, string> = { OK: '●', WARNING: '▲', FAULT: '■' };   // shape + text, never colour alone

  constructor() {
    this.api.sites().subscribe({
      next: (s) => this.sites.set(s),
      error: (e) => this.error.set(`Could not load sites (HTTP ${e.status})`),
    });
  }
}