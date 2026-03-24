import axios from "axios";

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

const post = async <T = any>(path: string, body?: unknown): Promise<T> => {
  try {
    const res = await axios.post(path, body);
    return res.data as T;
  } catch {
    const res = await axios.post(`http://localhost:8000${path}`, body);
    return res.data as T;
  }
};

const get = async <T = any>(path: string): Promise<T> => {
  try {
    const res = await axios.get(path);
    return res.data as T;
  } catch {
    const res = await axios.get(`http://localhost:8000${path}`);
    return res.data as T;
  }
};

export const saveScenarioApi = (payload: SaveScenarioPayload) => post("/api/scenario/save", payload);
export const listScenariosApi = () => get<{ items: any[] }>("/api/scenarios");
export const restoreScenarioApi = (id: string) => post(`/api/scenario/restore/${id}`);
export const deleteScenarioApi = (id: string) => post(`/api/scenario/delete/${id}`);
export const compareScenarioApi = (baseId: string, targetId: string) =>
  post("/api/scenario/compare", { base_scenario_id: baseId, target_scenario_id: targetId });

