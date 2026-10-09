-- Forecast Studio schema (PostgreSQL). Loaded automatically by the postgres
-- container on first start; dummy data is added by db/seed.py.

CREATE TABLE IF NOT EXISTS load (
    state        text             NOT NULL,
    date         date             NOT NULL,
    time_block   smallint         NOT NULL CHECK (time_block BETWEEN 1 AND 96),
    total_drawal double precision,
    PRIMARY KEY (state, date, time_block)
);

CREATE TABLE IF NOT EXISTS sldc (LIKE load INCLUDING ALL);

CREATE TABLE IF NOT EXISTS weather_mean (
    state             text     NOT NULL,
    date              date     NOT NULL,
    time_block        smallint NOT NULL CHECK (time_block BETWEEN 1 AND 96),
    temperature       double precision,
    humidity          double precision,
    precipitation     double precision,
    cloud_cover       double precision,
    cloud_cover_low   double precision,
    sunshine_duration double precision,
    direct_radiation  double precision,
    wind_speed_10m    double precision,
    PRIMARY KEY (state, date, time_block)
);

CREATE TABLE IF NOT EXISTS weather_loc (
    state             text     NOT NULL,
    date              date     NOT NULL,
    time_block        smallint NOT NULL CHECK (time_block BETWEEN 1 AND 96),
    location          text     NOT NULL,
    temperature       double precision,
    humidity          double precision,
    precipitation     double precision,
    cloud_cover       double precision,
    cloud_cover_low   double precision,
    sunshine_duration double precision,
    direct_radiation  double precision,
    wind_speed_10m    double precision,
    PRIMARY KEY (state, date, time_block, location)
);

CREATE TABLE IF NOT EXISTS forecast (
    state       text     NOT NULL,
    date        date     NOT NULL,
    block       smallint NOT NULL CHECK (block BETWEEN 1 AND 96),
    forecast_mw double precision,
    PRIMARY KEY (state, date, block)
);

CREATE TABLE IF NOT EXISTS sldc_forecast (
    state       text     NOT NULL,
    date        date     NOT NULL,
    time_block  smallint NOT NULL CHECK (time_block BETWEEN 1 AND 96),
    forecast_mw double precision,
    PRIMARY KEY (state, date, time_block)
);
