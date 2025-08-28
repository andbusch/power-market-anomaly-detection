"""
utility functions for fetching and processing ercot data

data used for model:

Zonal LMP
    - resolution: 5 min, by load zone
    - notes: filter by LZ_* for load zone data. needs to be aggregated to hourly level.
    - save path: data/ercot/zonal_lmp

Actual System Load by Weather Zone
    - resolution: hourly, forecast zone
    - notes: using weather zone, as more data is aggregated in this manner
    - save path: data/ercot/actual_sys_load

Seven-Day Load Forecast by Model and Weather Zone
    - resolution: hourly, forecast zone with seven day window
    - save path: data/ercot/seven_day_load_forecast

Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region
    - resolution: hourly, region
    - save path: data/ercot/wind_production

Solar Power Production - Actual 5-Minute Averaged Values by Geographical Region
    - resolution: hourly, region
    - save path: data/ercot/solar_production

"""

import torch
import datetime
from typing import List, Generator

def get_zonal_lmp(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Zonal LMP data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_zonal_lmp_features)
    """
    raise NotImplementedError

def get_actual_sys_load(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Actual System Load data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_actual_sys_load_features)
    """
    raise NotImplementedError

def get_seven_day_load_forecast(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Seven-Day Load Forecast data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_seven_day_load_forecast_features)
    """
    raise NotImplementedError

def get_wind_production(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Wind Production data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_wind_production_features)
    """
    raise NotImplementedError

def get_solar_production(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Solar Production data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_solar_production_features)
    """
    raise NotImplementedError

def process_daily_data(start_date: str, end_date: str, window_size: int, cache_data: bool) -> Generator[torch.Tensor, None, None]:
    """
    Generator that yields processed data per day

    """
    if window_size != 24:
        raise NotImplementedError("Only window size of 24 is supported.")
    
    start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start_date_dt

    while current_date <= end_date_dt:
        zonal_tensor = get_zonal_lmp(current_date, cache_data)
        actual_sys_load_tensor = get_actual_sys_load(current_date, cache_data)
        seven_day_load_forecast_tensor = get_seven_day_load_forecast(current_date, cache_data)
        wind_production_tensor = get_wind_production(current_date, cache_data)
        solar_production_tensor = get_solar_production(current_date, cache_data)

        # TODO: confirm dim=1 is correct
        yield torch.cat([zonal_tensor, actual_sys_load_tensor, seven_day_load_forecast_tensor, wind_production_tensor, solar_production_tensor], dim=1)

        current_date += datetime.timedelta(days=1)
    

def get_processed_ercot_data(start_date: str, end_date: str, window_size: int = 24, batch_size: int = 32, cache_data: bool = False) -> Generator[torch.Tensor, None, None]:
    """
    Generator that yields processed data in batches
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        window_size: Window size for processing
        batch_size: Batch size for processing
        cache_data: whether or not to save power market data to disk
    
    """
    daily_tensors : List[torch.Tensor] = []

    for day_tensor in process_daily_data(start_date, end_date, window_size, cache_data):  # Each: (1, window_size, num_features)
        daily_tensors.append(day_tensor)

        if len(daily_tensors) == batch_size:
            yield torch.cat(daily_tensors, dim=0)  # (batch_size, window_size, num_features)
            daily_tensors = []

    # Yield remainder
    if daily_tensors:
        yield torch.cat(daily_tensors, dim=0)  # (remainder_size, window_size, num_features)