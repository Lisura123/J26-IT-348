import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { API_URL } from './api';
import { Site, TimePoint } from './models';

@Injectable({ providedIn: 'root' })
export class TwinApi {
  private http = inject(HttpClient);

  sites() {
    return this.http.get<Site[]>(`${API_URL}/sites`);
  }

  timeseries(siteId: string) {
    return this.http.get<TimePoint[]>(`${API_URL}/sites/${siteId}/timeseries`);
  }
}