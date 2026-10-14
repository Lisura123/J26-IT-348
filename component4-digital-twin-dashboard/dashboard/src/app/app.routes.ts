import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';
import { Shell } from './layout/shell';
import { Login } from './pages/login/login';
import { Sites } from './pages/sites/sites';

export const routes: Routes = [
  { path: 'login', component: Login },
  {
    path: '',
    component: Shell,
    canActivate: [authGuard],
    canActivateChild: [authGuard],
    children: [
      { path: '', component: Sites },
      // lazy-loaded: Chart.js is only downloaded when a panel page is opened
      { path: 'sites/:id', loadComponent: () => import('./pages/site-detail/site-detail').then((m) => m.SiteDetail) },
      // Step 16: alerts (view_explanations), reports (view_reports)
    ],
  },
  { path: '**', redirectTo: '' },
];