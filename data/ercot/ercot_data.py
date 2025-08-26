"""
utility functions for fetching and processing ercot data

data used for model:

Zonal LMP
    - resolution: 5 min, by load zone
    - notes: filter by LZ_* for load zone data. needs to be aggregated to hourly level.

Actual Demand
    - resolution: hourly, forecast zone
Forecasted Demand
    - resolution: hourly, forecast zone
Wind production
    - resolution: hourly, region
Solar production
    - resolution: hourly, region

"""