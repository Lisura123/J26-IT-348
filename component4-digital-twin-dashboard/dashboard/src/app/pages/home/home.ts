import { HttpClient } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { API_URL } from '../../core/api';
import { AuthService } from '../../core/auth.service';

/** Step 14 placeholder: proves the token reaches the API. Replaced by the site overview in Step 15. */
@Component({
  selector: 'app-home',
  template: `
    <div class="card">
      <h2>Welcome, {{ auth.user()?.name }}</h2>
      <p>Role: <strong>{{ auth.user()?.role }}</strong></p>
      <p>Permissions ({{ auth.user()?.permissions?.length }}):</p>
      <ul>
        @for (p of auth.user()?.permissions; track p) { <li>{{ p }}</li> }
      </ul>
      <p>API test – <code>GET /sites</code>: <strong>{{ apiResult() }}</strong></p>
    </div>
  `,
})
export class Home {
  auth = inject(AuthService);
  private http = inject(HttpClient);
  apiResult = signal('loading…');

  constructor() {
    this.http.get<unknown[]>(`${API_URL}/sites`).subscribe({
      next: (sites) => this.apiResult.set(`OK, ${sites.length} sites received`),
      error: (e) => this.apiResult.set(`failed (HTTP ${e.status})`),
    });
  }
}