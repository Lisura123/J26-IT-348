export interface Site {
  site_id: string;
  panel_id: string;
  display_name: string;
  protocol: string;
  status: 'OK' | 'WARNING' | 'FAULT';
  open_alerts: number;
  worst_priority: string | null;
  latest?: {
    timestamp_utc: string;
    power_mw: number;
    expected_power_mw: number;
    power_band_low_mw: number;
    power_band_high_mw: number;
    physics_cause: string;
    within_expected_band: boolean;
  };
  model?: {                                   // only for roles with view_model_status
    model_version: string;
    latest_published_version: string;
    model_up_to_date: boolean;
    link_quality: string;
  };
}

export interface TimePoint {
  record_id: string;
  timestamp_utc: string;
  power_mw: number;
  expected_power_mw: number;
  power_band_low_mw: number;
  power_band_high_mw: number;
  within_expected_band: boolean;
  physics_cause: string;
  irradiance_wm2_est: number;
}