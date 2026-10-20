import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-login',
  imports: [FormsModule],
  template: `
    <div class="login-page">
      <form class="card" (ngSubmit)="submit()">
        <h1>Solar PV Digital Twin</h1>
        <p class="muted">Sign in to see how your solar panels are doing.</p>

        <label>Email
          <input name="email" type="email" [(ngModel)]="email" required autocomplete="username" />
        </label>
        <label>Password
          <input name="password" type="password" [(ngModel)]="password" required autocomplete="current-password" />
        </label>

        @if (error()) { <p class="error">{{ error() }}</p> }
        <button type="submit" [disabled]="loading()">{{ loading() ? 'Signing in…' : 'Sign in' }}</button>
      </form>
    </div>
  `,
  styles: `
    .login-page { min-height: 100vh; display: grid; place-items: center; padding: 16px; }
    .card { width: 100%; max-width: 360px; display: grid; gap: 14px; }
    h1 { margin: 0; font-size: 1.4rem; color: var(--green-700); }
    label { display: grid; gap: 4px; font-size: .9rem; }
    .error { color: var(--red-600); margin: 0; }
  `,
})
export class Login {
  private auth = inject(AuthService);
  private router = inject(Router);

  email = '';
  password = '';
  loading = signal(false);
  error = signal('');

  submit(): void {
    this.loading.set(true);
    this.error.set('');
    this.auth.login(this.email, this.password).subscribe({
      next: () => this.router.navigate(['/']),
      error: (e) => {
        this.error.set(e.status === 401
          ? 'Incorrect email or password. Please try again.'
          : 'Cannot connect to the server right now. Please try again in a moment.');
        this.loading.set(false);
      },
    });
  }
}