import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { API_URL } from './api';
import { Alert, AlertDetail, Metrics, Site, TimePoint } from './models';

@Injectable({ providedIn: 'root' })
export class TwinApi {
  private http = inject(HttpClient);

  sites() {
    return this.http.get<Site[]>(`${API_URL}/sites`);
  }

  timeseries(siteId: string) {
    return this.http.get<TimePoint[]>(`${API_URL}/sites/${siteId}/timeseries`);
  }

  alerts() {
    return this.http.get<Alert[]>(`${API_URL}/alerts`);
  }

  alert(alertId: string) {
    return this.http.get<AlertDetail>(`${API_URL}/alerts/${alertId}`);
  }

  metrics() {
    return this.http.get<Metrics>(`${API_URL}/metrics`);
  }
}