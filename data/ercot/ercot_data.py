"""
utility functions for fetching and processing ercot data

data used for model:

Zonal LMP
    - resolution: 5 min, by load zone
    - notes: filter by LZ_* for load zone data. needs to be aggregated to hourly level.
    - save path: data/ercot/zonal_lmp
    - https://apiexplorer.ercot.com/api-details#api=pubapi-apim-api&operation=getData_lmp_node_zone_hub

Actual System Load by Weather Zone
    - resolution: hourly, forecast zone
    - notes: using weather zone, as more data is aggregated in this manner
    - save path: data/ercot/actual_sys_load
    - https://apiexplorer.ercot.com/api-details#api=pubapi-apim-api&operation=getData_act_sys_load_by_wzn

Seven-Day Load Forecast by Model and Weather Zone
    - resolution: hourly, forecast zone with seven day window
    - save path: data/ercot/seven_day_load_forecast
    - https://apiexplorer.ercot.com/api-details#api=pubapi-apim-api&operation=getData_lf_by_model_weather_zone

Wind Power Production - Actual 5-Minute Averaged Values by Geographical Region
    - resolution: hourly, region
    - save path: data/ercot/wind_production
    - https://apiexplorer.ercot.com/api-details#api=pubapi-apim-api&operation=getData_wpp_actual_5min_avg_values_geo

Solar Power Production - Actual 5-Minute Averaged Values by Geographical Region
    - resolution: hourly, region
    - save path: data/ercot/solar_production
    - https://apiexplorer.ercot.com/api-details#api=pubapi-apim-api&operation=getData_spp_actual_5min_avg_values_geo

"""

import torch
import datetime
import os
import json
import requests
import pandas as pd
from typing import List, Generator, Optional
from pathlib import Path

ERCOT_BASE_URL = "https://api.ercot.com/api/public-reports"

def _get_access_token() -> str:
    """
    returns the application token saved in the environment or gets one
    """

    username = os.environ.get("ERCOT_USERNAME")
    password = os.environ.get("ERCOT_PASSWORD")

    if not username:
        raise ValueError("ERCOT_USERNAME environment variable is required")

    if not password:
        raise ValueError("ERCOT_PASSWORD environment variable is required")

    # Check if access token exists and is still valid
    existing_token = os.environ.get("ERCOT_ACCESS_TOKEN")
    expires_str = os.environ.get("ERCOT_ACCESS_TOKEN_EXPIRES_IN", "")
    expires_time = None

    if existing_token and expires_str:
        try:
            expires_time = datetime.datetime.fromisoformat(expires_str)
            if datetime.datetime.now() < expires_time:
                return existing_token
        except (ValueError, TypeError):
            pass  # Invalid format, fetch new token

    # Prepare authentication request data
    auth_data = {
        'username': username,
        'password': password,
        'grant_type': 'password',
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        'response_type': 'id_token'
    }

    auth_response = requests.post("https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token", data=auth_data)

    try:
        auth_response.raise_for_status()
    except Exception as e:
        print(f"Error in request. Response body: {auth_response.json()}")
        raise e

    access_token = auth_response.json().get("access_token")

    # Store token and expiration (60 minutes from now)
    os.environ["ERCOT_ACCESS_TOKEN"] = access_token
    os.environ["ERCOT_ACCESS_TOKEN_EXPIRES_IN"] = (datetime.datetime.now() + datetime.timedelta(minutes=60)).isoformat()

    return access_token


def _make_ercot_api_call(report_id: str, **params) -> Optional[dict]:
    """Make API call to ERCOT with authentication"""
    api_key = os.getenv('ERCOT_SUBSCRIPTION_KEY')
    if not api_key:
        raise ValueError("ERCOT_SUBSCRIPTION_KEY environment variable is required")

    access_token = _get_access_token()

    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Authorization': f'Bearer {access_token}'
    }
    
    url = f"{ERCOT_BASE_URL}/{report_id}"
    
    api_params = {}
    
    # Add any additional parameters
    api_params.update(params)
    
    try:
        response = requests.get(url, headers=headers, params=api_params)
        response.raise_for_status()
        json_data = response.json()
        
        # Save JSON response for debugging
        debug_dir = Path("debug_json_responses")
        debug_dir.mkdir(exist_ok=True)
        debug_file = debug_dir / f"{report_id.replace('/', '_')}.json"
        
        with open(debug_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved API response to {debug_file}")
        
        return json_data
    except requests.RequestException as e:
        print(f"Error fetching data from ERCOT API: {e}")
        return None

def _load_cached_data(cache_path: Path) -> Optional[torch.Tensor]:
    """Load cached tensor data"""
    if cache_path.exists():
        try:
            return torch.load(cache_path)
        except Exception as e:
            print(f"Error loading cached data from {cache_path}: {e}")
    return None

def _save_cached_data(cache_path: Path, tensor: torch.Tensor) -> None:
    """Save tensor data to cache"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, cache_path)
    except Exception as e:
        print(f"Error saving cached data to {cache_path}: {e}")

def _process_to_hourly_tensor(data: list, date: datetime.datetime, num_features: int) -> torch.Tensor:
    """Process API data into hourly tensor format"""
    if not data:
        # Return zeros if no data
        return torch.zeros(1, 24, num_features)
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data)
    
    # Create 24-hour tensor (assuming hourly data for now)
    # This is a simplified implementation - actual processing depends on data structure
    tensor_data = torch.zeros(1, 24, num_features)
    
    return tensor_data

def get_zonal_lmp(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Zonal LMP data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_zonal_lmp_features)
    """
    cache_dir = Path("data/ercot/zonal_lmp")
    cache_file = cache_dir / f"zonal_lmp_{date.strftime('%Y-%m-%d')}.pt"
    
    if cache_data:
        cached_tensor = _load_cached_data(cache_file)
        if cached_tensor is not None:
            return cached_tensor
    
    # add custom params for zonal_lmp
    start_datetime = date.strftime('%Y-%m-%dT00:00:00')
    end_datetime = date.strftime('%Y-%m-%dT23:59:59')
    params = {
        'SCEDTimestampFrom':start_datetime,
        'SCEDTimestampTo':end_datetime,
        'repeatHourFlag': False
    }

    api_data = _make_ercot_api_call("np6-788-cd/lmp_node_zone_hub", **params)
    
    if api_data is None:
        # Return default tensor if API call fails
        print('API data is None')
        tensor = torch.zeros(1, 24, 8)
    else:
        # Process API data to tensor
        data_list = api_data.get('data', [])
        
        if data_list:
            # Convert array format to DataFrame
            # Fields: SCEDTimestamp, repeatHourFlag, settlementPoint, LMP
            df = pd.DataFrame(data_list, columns=['SCEDTimestamp', 'repeatHourFlag', 'settlementPoint', 'LMP'])

            df.to_csv('debug_csv_responses/zonal_lmp.csv')
            return torch.zeros(1, 24, 8)
            
            # Filter for load zone data (LZ_*)
            lz_data = df[df['settlementPoint'].str.startswith('LZ_')]
            
            if not lz_data.empty:
                # Group by hour and settlement point, take mean
                lz_data['hour'] = pd.to_datetime(lz_data['SCEDTimestamp']).dt.hour
                grouped = lz_data.groupby(['hour', 'settlementPoint'])['LMP'].mean().unstack(fill_value=0)
                
                # Ensure we have 24 hours and pad/truncate to 8 zones
                hourly_data = torch.zeros(24, 8)
                for hour in range(24):
                    if hour in grouped.index:
                        zone_values = grouped.loc[hour].values[:8]  # Take first 8 zones
                        hourly_data[hour, :len(zone_values)] = torch.tensor(zone_values, dtype=torch.float32)
                
                tensor = hourly_data.unsqueeze(0)  # Add batch dimension
            else:
                tensor = torch.zeros(1, 24, 8)
        else:
            tensor = torch.zeros(1, 24, 8)
    
    # Save to cache if enabled
    if cache_data:
        _save_cached_data(cache_file, tensor)
    
    return tensor

def get_actual_sys_load(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Actual System Load data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_actual_sys_load_features)
    """
    cache_dir = Path("data/ercot/actual_sys_load")
    cache_file = cache_dir / f"actual_sys_load_{date.strftime('%Y-%m-%d')}.pt"
    
    # Check cache first if caching is enabled
    if cache_data:
        cached_tensor = _load_cached_data(cache_file)
        if cached_tensor is not None:
            return cached_tensor
        
    start_datetime = date.strftime('%Y-%m-%d')
    end_datetime = date.strftime('%Y-%m-%d')
        
    params = {
        'operatingDayFrom':start_datetime,
        'operatingDayTo':end_datetime
    }
    
    # Fetch from API
    api_data = _make_ercot_api_call("np6-345-cd/act_sys_load_by_wzn", **params)
    
    if api_data is None:
        # Return default tensor if API call fails
        print('API data is None')
        tensor = torch.zeros(1, 24, 8)
    else:
        # Process API data to tensor
        data_list = api_data.get('data', [])
        
        if data_list:
            df = pd.DataFrame(data_list)

            # TODO: remove
            df.to_csv('debug_csv_responses/sys_load.csv')
            return torch.zeros(1, 24, 8)

            if 'actual_system_load' in df.columns and 'weather_zone' in df.columns:
                # Group by hour and weather zone
                df['hour'] = pd.to_datetime(df['hour_ending']).dt.hour if 'hour_ending' in df.columns else pd.to_datetime(df['delivery_date']).dt.hour
                grouped = df.groupby(['hour', 'weather_zone'])['actual_system_load'].mean().unstack(fill_value=0)
                
                # Ensure we have 24 hours and pad/truncate to 8 weather zones
                hourly_data = torch.zeros(24, 8)
                for hour in range(24):
                    if hour in grouped.index:
                        zone_values = grouped.loc[hour].values[:8]  # Take first 8 zones
                        hourly_data[hour, :len(zone_values)] = torch.tensor(zone_values, dtype=torch.float32)
                
                tensor = hourly_data.unsqueeze(0)  # Add batch dimension
            else:
                tensor = torch.zeros(1, 24, 8)
        else:
            tensor = torch.zeros(1, 24, 8)
    
    # Save to cache if enabled
    if cache_data:
        _save_cached_data(cache_file, tensor)
    
    return tensor

def get_seven_day_load_forecast(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Seven-Day Load Forecast data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_seven_day_load_forecast_features)
    """
    cache_dir = Path("data/ercot/seven_day_load_forecast")
    cache_file = cache_dir / f"seven_day_load_forecast_{date.strftime('%Y-%m-%d')}.pt"
    
    # Check cache first if caching is enabled
    if cache_data:
        cached_tensor = _load_cached_data(cache_file)
        if cached_tensor is not None:
            return cached_tensor
        
    start_datetime = date.strftime('%Y-%m-%d')
    end_datetime = date.strftime('%Y-%m-%d')
    params = {
        'DSTFlag': False,
        'deliveryDateFrom': start_datetime,
        'deliveryDateTo': end_datetime
    }
    
    # Fetch from API
    api_data = _make_ercot_api_call("np3-565-cd/lf_by_model_weather_zone", **params)
    
    if api_data is None:
        # Return default tensor if API call fails - assuming multiple models and weather zones
        print('API data is None')
        tensor = torch.zeros(1, 24, 16)
    else:
        # Process API data to tensor
        data_list = api_data.get('data', [])
        
        if data_list:
            df = pd.DataFrame(data_list)

            # TODO: remove
            df.to_csv('debug_csv_responses/forecast_load.csv')
            return torch.zeros(1, 24, 8)

            if 'forecast_load' in df.columns and 'weather_zone' in df.columns and 'model_id' in df.columns:
                # Group by hour, weather zone, and model
                df['hour'] = pd.to_datetime(df['hour_ending']).dt.hour if 'hour_ending' in df.columns else pd.to_datetime(df['delivery_date']).dt.hour
                
                # Create a combined key for model and weather zone
                df['model_zone'] = df['model_id'].astype(str) + '_' + df['weather_zone'].astype(str)
                grouped = df.groupby(['hour', 'model_zone'])['forecast_load'].mean().unstack(fill_value=0)
                
                # Ensure we have 24 hours and pad/truncate to 16 features (assuming 2 models × 8 zones)
                hourly_data = torch.zeros(24, 16)
                for hour in range(24):
                    if hour in grouped.index:
                        feature_values = grouped.loc[hour].values[:16]  # Take first 16 features
                        hourly_data[hour, :len(feature_values)] = torch.tensor(feature_values, dtype=torch.float32)
                
                tensor = hourly_data.unsqueeze(0)  # Add batch dimension
            else:
                tensor = torch.zeros(1, 24, 16)
        else:
            tensor = torch.zeros(1, 24, 16)
    
    # Save to cache if enabled
    if cache_data:
        _save_cached_data(cache_file, tensor)
    
    return tensor

def get_wind_production(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Wind Production data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_wind_production_features)
    """
    cache_dir = Path("data/ercot/wind_production")
    cache_file = cache_dir / f"wind_production_{date.strftime('%Y-%m-%d')}.pt"
    
    # Check cache first if caching is enabled
    if cache_data:
        cached_tensor = _load_cached_data(cache_file)
        if cached_tensor is not None:
            return cached_tensor
        
    start_datetime = date.strftime('%Y-%m-%dT00:00:00')
    end_datetime = date.strftime('%Y-%m-%dT23:59:59')
    params = {
        'intervalEndingFrom':start_datetime,
        'intervalEndingTo':end_datetime
    }
    
    # Fetch from API
    api_data = _make_ercot_api_call("np4-743-cd/wpp_actual_5min_avg_values_geo", **params)
    
    if api_data is None:
        # Return default tensor if API call fails - assuming 4 geographical regions
        print('API data is None')
        tensor = torch.zeros(1, 24, 4)
    else:
        # Process API data to tensor
        data_list = api_data.get('data', [])
        
        if data_list:
            df = pd.DataFrame(data_list)

            # TODO: remove
            df.to_csv('debug_csv_responses/wind_production.csv')
            return torch.zeros(1, 24, 4)

            if 'mw' in df.columns and 'geographical_region' in df.columns:
                # Group by hour and geographical region, aggregating 5-minute data to hourly
                df['hour'] = pd.to_datetime(df['interval_ending']).dt.hour if 'interval_ending' in df.columns else pd.to_datetime(df['delivery_date']).dt.hour
                grouped = df.groupby(['hour', 'geographical_region'])['mw'].mean().unstack(fill_value=0)
                
                # Ensure we have 24 hours and pad/truncate to 4 regions
                hourly_data = torch.zeros(24, 4)
                for hour in range(24):
                    if hour in grouped.index:
                        region_values = grouped.loc[hour].values[:4]  # Take first 4 regions
                        hourly_data[hour, :len(region_values)] = torch.tensor(region_values, dtype=torch.float32)
                
                tensor = hourly_data.unsqueeze(0)  # Add batch dimension
            else:
                tensor = torch.zeros(1, 24, 4)
        else:
            tensor = torch.zeros(1, 24, 4)
    
    # Save to cache if enabled
    if cache_data:
        _save_cached_data(cache_file, tensor)
    
    return tensor

def get_solar_production(date: datetime.datetime, cache_data: bool) -> torch.Tensor:
    """
    Solar Production data

    Args:
        date: Datetime object

    Returns:
        Tensor of shape (1, window_size, num_solar_production_features)
    """
    cache_dir = Path("data/ercot/solar_production")
    cache_file = cache_dir / f"solar_production_{date.strftime('%Y-%m-%d')}.pt"
    
    # Check cache first if caching is enabled
    if cache_data:
        cached_tensor = _load_cached_data(cache_file)
        if cached_tensor is not None:
            return cached_tensor
        
    start_datetime = date.strftime('%Y-%m-%dT00:00:00')
    end_datetime = date.strftime('%Y-%m-%dT23:59:59')
    params = {
        'intervalEndingFrom':start_datetime,
        'intervalEndingTo':end_datetime
    }
    
    # Fetch from API
    api_data = _make_ercot_api_call("np4-746-cd/spp_actual_5min_avg_values_geo", **params)
    
    if api_data is None:
        # Return default tensor if API call fails - assuming 4 geographical regions
        print('API data is None')
        tensor = torch.zeros(1, 24, 4)
    else:
        # Process API data to tensor
        data_list = api_data.get('data', [])
        
        if data_list:
            df = pd.DataFrame(data_list)

            # TODO: remove
            df.to_csv('debug_csv_responses/solar_production.csv')
            return torch.zeros(1, 24, 4)

            if 'mw' in df.columns and 'geographical_region' in df.columns:
                # Group by hour and geographical region, aggregating 5-minute data to hourly
                df['hour'] = pd.to_datetime(df['interval_ending']).dt.hour if 'interval_ending' in df.columns else pd.to_datetime(df['delivery_date']).dt.hour
                grouped = df.groupby(['hour', 'geographical_region'])['mw'].mean().unstack(fill_value=0)
                
                # Ensure we have 24 hours and pad/truncate to 4 regions
                hourly_data = torch.zeros(24, 4)
                for hour in range(24):
                    if hour in grouped.index:
                        region_values = grouped.loc[hour].values[:4]  # Take first 4 regions
                        hourly_data[hour, :len(region_values)] = torch.tensor(region_values, dtype=torch.float32)
                
                tensor = hourly_data.unsqueeze(0)  # Add batch dimension
            else:
                tensor = torch.zeros(1, 24, 4)
        else:
            tensor = torch.zeros(1, 24, 4)
    
    # Save to cache if enabled
    if cache_data:
        _save_cached_data(cache_file, tensor)
    
    return tensor

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

        # Concatenate along feature dimension (dim=2)
        yield torch.cat([zonal_tensor, actual_sys_load_tensor, seven_day_load_forecast_tensor, wind_production_tensor, solar_production_tensor], dim=2)

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


# TODO: remove
if __name__ == "__main__":
    for t in get_processed_ercot_data("2024-01-01", "2024-01-01", window_size=24, batch_size=32, cache_data=False):
        print(t.shape)