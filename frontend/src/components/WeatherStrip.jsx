import React from 'react';
import { Thermometer, Droplets, CloudRain, Wind } from 'lucide-react';

const WeatherStrip = ({ dayAhead, live, fmt }) => {
  const coefs = dayAhead?.metadata?.weather_coefs || {};
  const deltas = dayAhead?.series?.weather_feature_deltas || {};
  const temps = deltas.temperature || [];
  const maxTempDelta =
    temps.length > 0 ? Math.max(...temps.map(Number).filter(Number.isFinite)) : 0;
  const precip = deltas.precipitation || [];
  const totalRain = precip.reduce((s, v) => s + Math.max(0, Number(v) || 0), 0);

  const tempCoef = coefs.temperature || coefs.temp || 0;
  const humCoef = coefs.humidity || 0;

  const meta = live?.metadata || {};
  const divergenceGate = meta?.weather_divergence_gate;

  return (
    <div className="weather-strip">
      <div className="ws-item">
        <Thermometer size={13} />
        <span className="ws-label">Temp ?</span>
        <span
          className={`ws-value ${maxTempDelta > 3 ? 'ws-danger' : maxTempDelta > 1.5 ? 'ws-warn' : ''}`}
        >
          {maxTempDelta > 0 ? '+' : ''}
          {maxTempDelta.toFixed(1)}°C
        </span>
        {tempCoef ? <span className="ws-sub">{Math.abs(tempCoef).toFixed(0)} MW/°C</span> : null}
      </div>

      <div className="ws-item">
        <Droplets size={13} />
        <span className="ws-label">Humidity</span>
        <span className="ws-value">{Math.abs(humCoef).toFixed(1)} MW/%RH</span>
      </div>

      <div className="ws-item">
        <CloudRain size={13} />
        <span className="ws-label">Rain</span>
        <span
          className={`ws-value ${totalRain > 30 ? 'ws-danger' : totalRain > 10 ? 'ws-warn' : ''}`}
        >
          {totalRain.toFixed(0)}mm
        </span>
      </div>

      {divergenceGate !== undefined && (
        <div className="ws-item">
          <Wind size={13} />
          <span className="ws-label">Wx Gate</span>
          <span className="ws-value">{(divergenceGate * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
};

export default WeatherStrip;
