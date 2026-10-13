import { HttpClient } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';
import { API_URL } from './api';

export interface User {
  user_id: string;
  name: string;
  role: 'ADMIN' | 'ENGINEER' | 'MANAGER';
  permissions: string[];
}

interface LoginResponse {
  access_token: string;
  user: User;
}

const TOKEN_KEY = 'm4_token';
const USER_KEY = 'm4_user';

/**
 * Holds the logged-in user and JWT.
 * sessionStorage (not localStorage): the session ends when the browser tab closes.
 * UI permission checks here only hide things - the API enforces them again (403).
 */
@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private router = inject(Router);

  readonly user = signal<User | null>(this.readUser());
  readonly isLoggedIn = computed(() => this.user() !== null);

  get token(): string | null {
    return sessionStorage.getItem(TOKEN_KEY);
  }

  login(email: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${API_URL}/auth/login`, { email, password }).pipe(
      tap((res) => {
        sessionStorage.setItem(TOKEN_KEY, res.access_token);
        sessionStorage.setItem(USER_KEY, JSON.stringify(res.user));
        this.user.set(res.user);
      }),
    );
  }

  logout(): void {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
    this.user.set(null);
    this.router.navigate(['/login']);
  }

  can(permission: string): boolean {
    return this.user()?.permissions.includes(permission) ?? false;
  }

  private readUser(): User | null {
    const raw = sessionStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  }
}