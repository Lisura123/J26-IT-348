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


export interface Alert {
  alert_id: string;
  site_id: string;
  physics_cause: string;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  start_utc: string;
  end_utc: string;
  n_readings: number;
  mean_power_loss_pct: number;
  rule_id: string;
  recommended_action: string;
  assigned_role: string;
  respond_by_utc: string;
  tinyml_majority: string;
  tinyml_agree_frac: number;
}

export interface ShapExplanation {
  record_id: string;
  model_version: string;
  predicted_fault: string;
  confidence: number;
  base_value: number;
  features: { feature: string; value: number }[];   // sorted by |value|, largest first
}

export interface AlertDetail extends Alert {
  device_id: string;
  first_record_id: string;
  explanation: string;
  trigger: string;
  target_response_h: number;
  shap: ShapExplanation | null;                      // null when the role lacks view_shap
  shap_visible: boolean;
}

export interface Metrics {
  thermal: { u0: number; u1: number; rmse_with_wind_c: number; rmse_no_wind_c: number; wind_improvement_pct: number };
  power: { rmse_mw: number; mape_pct: number; r2: number; ablation_rmse_mw: number };
  conformal: { target: number; random_split_coverage: number; temporal_split_coverage: number };
  shap_vs_member2: { median_row_correlation: number; sign_agreement: number; top1_feature_agreement: number };
}