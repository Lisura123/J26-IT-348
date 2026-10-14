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
      // lazy-loaded pages: Chart.js is only downloaded when one of them is opened
      { path: 'sites/:id', loadComponent: () => import('./pages/site-detail/site-detail').then((m) => m.SiteDetail) },
      { path: 'alerts', data: { permission: 'view_explanations' },
        loadComponent: () => import('./pages/alerts/alerts').then((m) => m.Alerts) },
      { path: 'alerts/:id', data: { permission: 'view_explanations' },
        loadComponent: () => import('./pages/alert-detail/alert-detail').then((m) => m.AlertDetailPage) },
      { path: 'reports', data: { permission: 'view_reports' },
        loadComponent: () => import('./pages/reports/reports').then((m) => m.Reports) },
    ],
  },
  { path: '**', redirectTo: '' },
];