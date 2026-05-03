import React, { useState } from 'react';
import { Thermometer, Activity, CloudRain, Wind } from 'lucide-react';
import IntradayWeather from './IntradayWeather';
import PeakWindowStress from './PeakWindowStress';
import RainCloud from './RainCloud';
import Sensitivity from './Sensitivity';

export default function WeatherAnalysisPanel({
  data,
  dayAheadSeries,
  weatherAdj,
  setWeatherAdj,
  metricsSlot,
}) {
  const [activeTab, setActiveTab] = useState('intraday');

  if (!data) return <div className="muted p-4">No weather analysis data available.</div>;

  const tabs = [
    { id: 'intraday', label: 'Weather vs Normal', icon: Thermometer },
    { id: 'peak', label: 'Peak Load Stress', icon: Activity },
    { id: 'rain', label: 'Rain & Cloud', icon: CloudRain },
    { id: 'sensitivity', label: 'Load Attribution', icon: Wind },
  ];

  return (
    <div className="analysis-layout--full">
      {/* 1. Tabs first */}
      <div className="decision-tabs weather-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-btn ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* 2. Metrics cards (passed from parent) */}
      {metricsSlot}

      {/* 3. Content Panel */}
      <div className="analysis-main analysis-main--stretch">
        <div className="flex-1 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
            {activeTab === 'intraday' && data.intraday && <IntradayWeather data={data.intraday} />}
            {activeTab === 'peak' && data.peak_windows && (
              <PeakWindowStress data={data.peak_windows} intraday={data.intraday} />
            )}
            {activeTab === 'rain' && data.rain_metrics && (
              <RainCloud data={data.rain_metrics} intraday={data.intraday} />
            )}
            {activeTab === 'sensitivity' && data.sensitivity && (
              <Sensitivity
                data={data.sensitivity}
                series={dayAheadSeries}
                weatherAdj={weatherAdj}
                setWeatherAdj={setWeatherAdj}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
