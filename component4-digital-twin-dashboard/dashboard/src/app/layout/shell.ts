import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from '../core/auth.service';

/** Top bar + navigation. Links appear only if the role holds the permission. */
@Component({
  selector: 'app-shell',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <header class="topbar">
      <strong>☀ PV Digital Twin</strong>
      <nav>
        <a routerLink="/" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }">Sites</a>
        @if (auth.can('view_explanations')) { <a routerLink="/alerts" routerLinkActive="active">Alerts</a> }
        @if (auth.can('view_reports')) { <a routerLink="/reports" routerLinkActive="active">Reports</a> }
      </nav>
      <span class="user">{{ auth.user()?.name }} <span class="badge">{{ auth.user()?.role }}</span></span>
      <button class="link" (click)="auth.logout()">Log out</button>
    </header>
    <main><router-outlet /></main>
  `,
  styles: `
    .topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; padding: 12px 16px;
              background: var(--green-700); color: #fff; }
    nav { display: flex; gap: 12px; flex: 1; }
    nav a { color: #e8f5e9; text-decoration: none; padding: 4px 8px; border-radius: 6px; }
    nav a.active { background: rgba(255,255,255,.18); color: #fff; }
    .badge { font-size: .7rem; background: #fff; color: var(--green-700); padding: 2px 6px; border-radius: 4px; }
    .link { background: none; border: 1px solid rgba(255,255,255,.5); color: #fff; }
    main { padding: 16px; max-width: 1200px; margin: 0 auto; }
  `,
})
export class Shell {
  auth = inject(AuthService);
}