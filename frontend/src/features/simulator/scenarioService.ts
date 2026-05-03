import axios from 'axios';
import { getApiUrl } from '../../apiConfig';

type SaveScenarioPayload = {
  scenario_name: string;
  created_by: string;
  notes: string;
  drivers_applied: Record<string, boolean>;
  block_adjustments: unknown;
  forecast_summary: {
    peak_mw: number;
    energy_mu: number;
    net_impact_pct: number;
  };
  weather_inputs: Record<string, unknown>;
  holiday_template: Record<string, unknown>;
  forecast_output: Record<string, unknown>;
};

export const saveScenarioApi = (payload: SaveScenarioPayload) =>
  axios.post(getApiUrl('/scenario/save'), payload).then((r) => r.data);
export const listScenariosApi = () => axios.get(getApiUrl('/scenarios')).then((r) => r.data);
export const restoreScenarioApi = (id: string) =>
  axios.post(getApiUrl(`/scenario/restore/${id}`)).then((r) => r.data);
export const deleteScenarioApi = (id: string) =>
  axios.post(getApiUrl(`/scenario/delete/${id}`)).then((r) => r.data);
export const compareScenarioApi = (baseId: string, targetId: string) =>
  axios
    .post(getApiUrl('/scenario/compare'), {
      base_scenario_id: baseId,
      target_scenario_id: targetId,
    })
    .then((r) => r.data);
