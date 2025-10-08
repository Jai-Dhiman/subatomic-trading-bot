-- Create battery data tables for all 11 houses
-- Run this SQL in Supabase SQL Editor before uploading battery data

-- House 1
CREATE TABLE IF NOT EXISTS house1_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house1_battery_timestamp ON house1_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house1_battery_house_id ON house1_battery(house_id);

-- House 2
CREATE TABLE IF NOT EXISTS house2_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house2_battery_timestamp ON house2_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house2_battery_house_id ON house2_battery(house_id);

-- House 3
CREATE TABLE IF NOT EXISTS house3_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house3_battery_timestamp ON house3_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house3_battery_house_id ON house3_battery(house_id);

-- House 4
CREATE TABLE IF NOT EXISTS house4_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house4_battery_timestamp ON house4_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house4_battery_house_id ON house4_battery(house_id);

-- House 5
CREATE TABLE IF NOT EXISTS house5_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house5_battery_timestamp ON house5_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house5_battery_house_id ON house5_battery(house_id);

-- House 6
CREATE TABLE IF NOT EXISTS house6_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house6_battery_timestamp ON house6_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house6_battery_house_id ON house6_battery(house_id);

-- House 7
CREATE TABLE IF NOT EXISTS house7_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house7_battery_timestamp ON house7_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house7_battery_house_id ON house7_battery(house_id);

-- House 8
CREATE TABLE IF NOT EXISTS house8_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house8_battery_timestamp ON house8_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house8_battery_house_id ON house8_battery(house_id);

-- House 9
CREATE TABLE IF NOT EXISTS house9_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house9_battery_timestamp ON house9_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house9_battery_house_id ON house9_battery(house_id);

-- House 10 (2 batteries - 80 kWh)
CREATE TABLE IF NOT EXISTS house10_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house10_battery_timestamp ON house10_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house10_battery_house_id ON house10_battery(house_id);

-- House 11 (2 batteries - 80 kWh)
CREATE TABLE IF NOT EXISTS house11_battery (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    house_id INTEGER NOT NULL,
    battery_soc_percent REAL NOT NULL,
    battery_charge_kwh REAL NOT NULL,
    battery_available_kwh REAL NOT NULL,
    battery_capacity_remaining_kwh REAL NOT NULL,
    battery_soh_percent REAL NOT NULL,
    battery_count INTEGER NOT NULL,
    total_capacity_kwh REAL NOT NULL,
    max_charge_rate_kw REAL NOT NULL,
    max_discharge_rate_kw REAL NOT NULL,
    action TEXT,
    trade_amount_kwh REAL,
    price_per_kwh REAL,
    consumption_kwh REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_house11_battery_timestamp ON house11_battery(timestamp);
CREATE INDEX IF NOT EXISTS idx_house11_battery_house_id ON house11_battery(house_id);

-- Verify tables created
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'house%_battery'
ORDER BY tablename;
