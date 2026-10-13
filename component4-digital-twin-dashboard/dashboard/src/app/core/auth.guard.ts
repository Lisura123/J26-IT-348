import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './auth.service';

/**
 * Route guard: user must be logged in, and must hold route.data.permission if one is set.
 * Example route: { path: 'alerts', data: { permission: 'view_explanations' }, ... }
 */
export const authGuard: CanActivateFn = (route) => {
  const auth = inject(AuthService);
  const router = inject(Router);
  if (!auth.isLoggedIn()) return router.createUrlTree(['/login']);
  const needed = route.data?.['permission'] as string | undefined;
  if (needed && !auth.can(needed)) return router.createUrlTree(['/']);
  return true;
};