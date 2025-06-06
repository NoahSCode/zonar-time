import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import altair as alt
import requests
import streamlit.components.v1 as components
from scipy.stats import linregress
from dataclasses import dataclass, field # Import dataclass for state management

################################################################################
# 0) NEW: Data Class for State Management
################################################################################
@dataclass
class BusState:
    """
    A dataclass to hold the state of a single bus during chronological processing.
    This acts as the 'value' in our bus_id -> state hash table.
    """
    # Trip state flags
    inside_origin: bool = False
    inside_dest: bool = False

    # Key timestamps for the current in-progress trip
    origin_arrival_time: pd.Timestamp = pd.NaT
    origin_depart_time: pd.Timestamp = pd.NaT
    dest_arrival_time: pd.Timestamp = pd.NaT

    # Data from the *previous* ping for this specific bus, needed for transitions
    prev_datetime: pd.Timestamp = pd.NaT
    prev_in_origin_zone: bool = False
    prev_in_dest_zone: bool = False
    prev_dist_traveled: float = np.nan

    # To track the start of the trip for distance calculation
    origin_depart_start_dist: float = np.nan
    
    # Store the actual last ping time inside the origin for more accurate map plotting
    last_ping_in_origin_dt_current_trip: pd.Timestamp = pd.NaT


################################################################################
# 1) Utility: Haversine & Circle checks (Unchanged)
################################################################################
EARTH_RADIUS_FEET = 20902231  # Earth's approximate radius in feet

def haversine_distance_vectorized(lat1_arr: np.ndarray, lon1_arr: np.ndarray, lat2_scalar: float, lon2_scalar: float) -> np.ndarray:
    """
    Calculate Haversine distance between arrays of points and a single point.
    Optimized to use NumPy arrays directly.
    """
    lat1_rad = np.radians(lat1_arr)
    lon1_rad = np.radians(lon1_arr)
    lat2_rad = np.radians(lat2_scalar)
    lon2_rad = np.radians(lon2_scalar)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_FEET * c

def interpolate_time(t1: pd.Timestamp, t2: pd.Timestamp) -> pd.Timestamp | None:
    """
    Interpolate the midpoint time between two timestamps.
    Returns None if either timestamp is NaT or None.
    """
    if pd.isna(t1) or pd.isna(t2):
        return None
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)
    midpoint = t1 + (t2 - t1) / 2
    return midpoint.replace(microsecond=0)

################################################################################
# 2) OPTIMIZED Ping-Based Stop Crossing Analysis (Hash Table Implementation)
################################################################################
def analyze_stop_crossings_hashtable(
    stops_df: pd.DataFrame, path_df: pd.DataFrame, origin_stop: str, destination_stop: str,
    radius_feet_origin: float = 200, radius_feet_destination: float = 200,
    op_day_cutoff_hour: int = 3
) -> pd.DataFrame:
    """
    Analyzes stop crossings using a single chronological pass over all pings,
    managing state for each bus in a hash table (dictionary). This avoids
    the overhead of pandas groupby and is more scalable.
    """
    if stops_df.empty or path_df.empty:
        return pd.DataFrame()

    try:
        origin_row = stops_df[stops_df["Stop Name"] == origin_stop].iloc[0]
        dest_row = stops_df[stops_df["Stop Name"] == destination_stop].iloc[0]
    except IndexError:
        st.error(f"Origin '{origin_stop}' or Destination '{destination_stop}' not found in stops data.")
        return pd.DataFrame()

    # --- Pre-computation Step ---
    # 1. Sort the entire dataframe by time. This is the key to the single-pass approach.
    df = path_df.sort_values("DateTime").copy()

    # 2. Vectorized calculation of distances and zone checks for all pings at once.
    df['in_origin_zone'] = haversine_distance_vectorized(df["Lat"].values, df["Lon"].values, origin_row["Lat"], origin_row["Lon"]) <= radius_feet_origin
    df['in_dest_zone'] = haversine_distance_vectorized(df["Lat"].values, df["Lon"].values, dest_row["Lat"], dest_row["Lon"]) <= radius_feet_destination

    # --- Hash Table and Iteration Step ---
    bus_states = {}  # The hash table: {bus_id: BusState}
    all_results_list = []

    # Iterate through each ping in chronological order. itertuples is much faster than iterrows.
    for ping in df.itertuples():
        bus_id = ping._asdict().get("Asset No.") # Use ._asdict() for robust column name access
        curr_datetime = ping.DateTime
        curr_in_origin = ping.in_origin_zone
        curr_in_dest = ping.in_dest_zone
        curr_dist_traveled = getattr(ping, "Distance Traveled(Miles)", np.nan)

        # Get the state for the current bus. If it's the first time we see this bus, create a new state.
        state = bus_states.get(bus_id)
        if state is None:
            state = BusState()
            bus_states[bus_id] = state

        # --- State Machine Logic (applied to each ping) ---
        # A bus has entered the origin zone for the first time on a trip.
        if not state.inside_origin and curr_in_origin:
            state.inside_origin = True
            state.origin_arrival_time = curr_datetime
        
        # A bus has just left the origin zone. This marks the departure.
        # This requires knowing the state of the *previous* ping for this bus.
        if state.inside_origin and state.prev_in_origin_zone and not curr_in_origin:
            state.inside_origin = False
            state.origin_depart_time = interpolate_time(state.prev_datetime, curr_datetime)
            # Store the distance reading at the start of the trip segment
            state.origin_depart_start_dist = state.prev_dist_traveled
            # Keep the last *actual* ping time inside the zone for map visualization
            state.last_ping_in_origin_dt_current_trip = state.prev_datetime

        # A bus has entered the destination zone for the first time *after* departing the origin.
        if not pd.isna(state.origin_depart_time) and not state.inside_dest and curr_in_dest:
            # Ignore entries into destination that happen before the official origin departure time
            if curr_datetime < state.origin_depart_time:
                # Update previous state and continue to the next ping
                state.prev_datetime = curr_datetime
                state.prev_in_origin_zone = curr_in_origin
                state.prev_in_dest_zone = curr_in_dest
                state.prev_dist_traveled = curr_dist_traveled
                continue

            state.inside_dest = True
            state.dest_arrival_time = curr_datetime
            
            # Calculate metrics for the completed trip leg
            actual_distance = 0.0
            if pd.notna(state.origin_depart_start_dist) and pd.notna(curr_dist_traveled):
                actual_distance = curr_dist_traveled - state.origin_depart_start_dist
            
            origin_idle_mins = (state.origin_depart_time - state.origin_arrival_time).total_seconds() / 60.0
            travel_time_mins = (state.dest_arrival_time - state.origin_depart_time).total_seconds() / 60.0

            # Assign to an operational day
            travel_date_str = None
            if pd.notnull(state.origin_depart_time):
                event_dt = state.origin_depart_time
                travel_date_to_format = event_dt.date() - timedelta(days=1) if event_dt.hour < op_day_cutoff_hour else event_dt.date()
                travel_date_str = travel_date_to_format.strftime("%m/%d/%Y")

            # Append the result
            all_results_list.append({
                "Vehicle no.": bus_id, "Travel Date": travel_date_str,
                "Origin Stop": origin_stop, "Destination Stop": destination_stop,
                "Origin Stop Arrival": state.origin_arrival_time,
                "Origin Stop Departure": state.origin_depart_time,
                "Last Ping In Origin DateTime": state.last_ping_in_origin_dt_current_trip,
                "Origin Stop Idle (mins)": round(origin_idle_mins, 2),
                "Destination Stop Entry": state.dest_arrival_time,
                "Destination Stop Departure": pd.NaT, # To be filled in later
                "Destination Stop Idle (mins)": None,
                "Actual Distance (miles)": round(actual_distance, 3) if actual_distance >= 0 else 0,
                "Travel Time": round(travel_time_mins, 2) if travel_time_mins >=0 else 0
            })

        # A bus has just left the destination zone, completing the full stop.
        if state.inside_dest and state.prev_in_dest_zone and not curr_in_dest:
            state.inside_dest = False
            dest_depart_time_val = interpolate_time(state.prev_datetime, curr_datetime)

            # Find the last record for this bus and update its departure/idle time.
            if all_results_list:
                last_record = all_results_list[-1]
                if (last_record["Vehicle no."] == bus_id and
                    pd.isna(last_record["Destination Stop Departure"]) and
                    last_record["Destination Stop Entry"] == state.dest_arrival_time):
                    
                    last_record["Destination Stop Departure"] = dest_depart_time_val
                    dest_idle_mins = (dest_depart_time_val - state.dest_arrival_time).total_seconds() / 60.0
                    last_record["Destination Stop Idle (mins)"] = round(dest_idle_mins, 2)

            # Reset the state for this bus to be ready for a new trip from origin->dest.
            state = BusState()
            bus_states[bus_id] = state

        # --- Update State ---
        # At the end of the loop for each ping, update the 'previous' state holder.
        state.prev_datetime = curr_datetime
        state.prev_in_origin_zone = curr_in_origin
        state.prev_in_dest_zone = curr_in_dest
        state.prev_dist_traveled = curr_dist_traveled
        # If the bus is currently in the origin, keep updating the last known ping time inside.
        if state.inside_origin and curr_in_origin:
             state.last_ping_in_origin_dt_current_trip = curr_datetime

    # --- Finalization Step ---
    if not all_results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results_list)
    datetime_cols = [
        "Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime",
        "Destination Stop Entry", "Destination Stop Departure"
    ]
    for col in datetime_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_datetime(results_df[col], errors='coerce')
    
    final_cols_ordered = [
        "Vehicle no.", "Travel Date", "Origin Stop", "Destination Stop",
        "Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime",
        "Origin Stop Idle (mins)",
        "Destination Stop Entry", "Destination Stop Departure", "Destination Stop Idle (mins)",
        "Actual Distance (miles)", "Travel Time"
    ]
    return results_df[[col for col in final_cols_ordered if col in results_df.columns]]


################################################################################
# 3) Distance Filter, 4) Idle Largest Cluster, 5) Idle IQR (Unchanged)
################################################################################
def filter_by_gap_clustering_largest(df: pd.DataFrame, gap_threshold: float = 0.3) -> pd.DataFrame:
    if df.empty or "Actual Distance (miles)" not in df.columns:
        return df.copy() 
    
    dist_col = "Actual Distance (miles)"
    sorted_df = df.dropna(subset=[dist_col]).sort_values(by=dist_col)
    
    if sorted_df.empty or len(sorted_df) < 2: 
        return df.copy() 

    distances = sorted_df[dist_col].values
    clusters_indices = [] 
    current_cluster_original_indices = [sorted_df.index[0]] 

    if len(distances) == 1: 
         clusters_indices.append(current_cluster_original_indices)
    else:
        for i in range(len(distances) - 1):
            if (distances[i+1] - distances[i]) >= gap_threshold:
                clusters_indices.append(current_cluster_original_indices)
                current_cluster_original_indices = [sorted_df.index[i+1]]
            else:
                current_cluster_original_indices.append(sorted_df.index[i+1])
        if current_cluster_original_indices: 
            clusters_indices.append(current_cluster_original_indices)

    if not clusters_indices:
        return pd.DataFrame(columns=df.columns) 

    largest_cluster_indices = max(clusters_indices, key=len)
    return df.loc[largest_cluster_indices].copy()


def filter_idle_by_gap_clustering_largest(df: pd.DataFrame, idle_col: str, gap_threshold: float = 0.3) -> pd.DataFrame:
    if df.empty or idle_col not in df.columns:
        return df.copy()

    sub_df = df.dropna(subset=[idle_col])
    if sub_df.empty or len(sub_df) < 2: 
        return df.copy() 

    sorted_sub = sub_df.sort_values(by=idle_col)
    idle_vals = sorted_sub[idle_col].values
    clusters_indices = []
    current_cluster_original_indices = [sorted_sub.index[0]]

    if len(idle_vals) == 1:
        clusters_indices.append(current_cluster_original_indices)
    else:
        for i in range(len(idle_vals) - 1):
            if (idle_vals[i+1] - idle_vals[i]) >= gap_threshold:
                clusters_indices.append(current_cluster_original_indices)
                current_cluster_original_indices = [sorted_sub.index[i+1]]
            else:
                current_cluster_original_indices.append(sorted_sub.index[i+1])
        if current_cluster_original_indices:
            clusters_indices.append(current_cluster_original_indices)
            
    if not clusters_indices:
        return pd.DataFrame(columns=df.columns)

    largest_cluster_indices = max(clusters_indices, key=len)
    return df.loc[largest_cluster_indices].copy()

def filter_idle_by_iqr(df: pd.DataFrame, idle_col: str, multiplier: float = 1.5) -> pd.DataFrame:
    if df.empty or idle_col not in df.columns:
        return df.copy()

    sub = df.dropna(subset=[idle_col])
    if sub.empty:
        return df.copy() 

    q1 = sub[idle_col].quantile(0.25)
    q3 = sub[idle_col].quantile(0.75)
    iqr = q3 - q1
    
    upper_cut = q3 if iqr == 0 else q3 + multiplier * iqr
    
    return df[(df[idle_col] <= upper_cut) | df[idle_col].isna()].copy()

################################################################################
# 6) Snap-to-Roads (Unchanged)
################################################################################
def snap_to_roads(coords: list[tuple[float, float]], api_key: str) -> list[tuple[float, float]]:
    if not coords: return []
    snapped_points = []
    BATCH_SIZE = 100 
    for i in range(0, len(coords), BATCH_SIZE):
        chunk = coords[i:i + BATCH_SIZE]
        path_param = "|".join(f"{c[0]},{c[1]}" for c in chunk)
        url = f"https://roads.googleapis.com/v1/snapToRoads?path={path_param}&interpolate=false&key={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status() 
            data = resp.json()
            for p in data.get("snappedPoints", []):
                snapped_points.append((p["location"]["latitude"], p["location"]["longitude"]))
        except requests.exceptions.RequestException as e:
            st.error(f"Snap to Roads API error: {e}")
            return [] 
        except Exception as e: 
            st.error(f"Error processing Snap to Roads response: {e}")
            return []
    return snapped_points

def embed_snapped_polyline_map(
    original_coords: list, snapped_coords: list,
    origin_stop_lat: float, origin_stop_lon: float, dest_stop_lat: float, dest_stop_lon: float,
    radius_feet_origin: float, radius_feet_destination: float, api_key: str
):
    if not original_coords and not snapped_coords:
        st.write("No coordinates to display on map.")
        return

    radius_meters_origin = float(radius_feet_origin) * 0.3048
    radius_meters_dest = float(radius_feet_destination) * 0.3048
    
    center_lat, center_lon = (origin_stop_lat, origin_stop_lon) 
    if snapped_coords:
        center_lat, center_lon = snapped_coords[0]
    elif original_coords:
        center_lat, center_lon = original_coords[0]

    snapped_js = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in snapped_coords)
    orig_js = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in original_coords)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
        <style>
            #map {{ height: 100%; width: 100%; }}
            html, body {{ margin: 0; padding: 0; height: 100%; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
        function initMap() {{
            var map = new google.maps.Map(document.getElementById('map'), {{
                zoom: 13,
                center: {{ lat: {center_lat}, lng: {center_lon} }}
            }});

            new google.maps.Circle({{
                strokeColor: "#0000FF", strokeOpacity: 0.8, strokeWeight: 2,
                fillColor: "#0000FF", fillOpacity: 0.1, map: map,
                center: {{ lat: {origin_stop_lat}, lng: {origin_stop_lon} }},
                radius: {radius_meters_origin}
            }});
            new google.maps.Marker({{
                position: {{ lat: {origin_stop_lat}, lng: {origin_stop_lon} }}, map: map, title: "Origin Stop"
            }});

            new google.maps.Circle({{
                strokeColor: "#00FF00", strokeOpacity: 0.8, strokeWeight: 2,
                fillColor: "#00FF00", fillOpacity: 0.1, map: map,
                center: {{ lat: {dest_stop_lat}, lng: {dest_stop_lon} }},
                radius: {radius_meters_dest}
            }});
            new google.maps.Marker({{
                position: {{ lat: {dest_stop_lat}, lng: {dest_stop_lon} }}, map: map, title: "Destination Stop"
            }});

            var snappedCoords = [{snapped_js}];
            if (snappedCoords.length > 1) {{
                var snappedRoute = new google.maps.Polyline({{
                    path: snappedCoords, geodesic: true, strokeColor: "#4285F4",
                    strokeOpacity: 1.0, strokeWeight: 4
                }});
                snappedRoute.setMap(map);
            }}

            var originalCoords = [{orig_js}];
            originalCoords.forEach(function(coord) {{
                new google.maps.Marker({{
                    position: coord, map: map,
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE, scale: 2,
                        fillColor: '#FF0000', fillOpacity: 0.7,
                        strokeColor: '#FF0000', strokeWeight: 1
                    }}
                }});
            }});
        }}
        </script>
        <script async src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap"></script>
    </body>
    </html>
    """
    components.html(html_code, height=600)

################################################################################
# 7) Main App (and helper functions)
################################################################################

def calculate_time_of_day_adjusted_vectorized(s_depart_dt: pd.Series, s_travel_date_dt: pd.Series) -> pd.Series:
    """
    Vectorized calculation of time of day, adjusting for trips past midnight.
    s_depart_dt: Series of departure datetimes (Timestamp)
    s_travel_date_dt: Series of operational travel dates (date objects)
    """
    if s_depart_dt.empty or s_travel_date_dt.empty:
        return pd.Series(index=s_depart_dt.index, dtype=float)

    depart_dt = pd.to_datetime(s_depart_dt, errors='coerce')
    
    time_val = depart_dt.dt.hour + depart_dt.dt.minute / 60.0

    travel_date_dt_ts = pd.to_datetime(s_travel_date_dt) 
    departure_date_only = depart_dt.dt.date
    
    is_one_day_after = (departure_date_only > s_travel_date_dt) & \
                       (departure_date_only <= (s_travel_date_dt + pd.to_timedelta(1, unit='D')))
                       
    is_two_days_after = (departure_date_only > (s_travel_date_dt + pd.to_timedelta(1, unit='D'))) & \
                        (departure_date_only <= (s_travel_date_dt + pd.to_timedelta(2, unit='D')))
    
    time_val = np.where(is_one_day_after, time_val + 24.0, time_val)
    time_val = np.where(is_two_days_after, time_val + 48.0, time_val)

    time_val[depart_dt.isna() | travel_date_dt_ts.isna()] = np.nan
    return pd.Series(time_val, index=s_depart_dt.index)


def format_time_of_day_label(time_val_24hr: float) -> str:
    """Converts a 24-hour+ numeric time to a 12-hour string with AM/PM."""
    if pd.isna(time_val_24hr):
        return "N/A"
    
    day_offset = int(time_val_24hr // 24)
    hour_of_day_24 = time_val_24hr % 24
    
    hour_int = int(hour_of_day_24)
    minute_fraction = hour_of_day_24 - hour_int
    minute_int = int(round(minute_fraction * 60))

    if minute_int == 60: 
        hour_int += 1
        minute_int = 0
        if hour_int == 24: 
            hour_int = 0
            day_offset += 1 

    period = "AM"
    hour_12 = hour_int
    if hour_12 == 0: 
        hour_12 = 12
    elif hour_12 == 12: 
        period = "PM"
    elif hour_12 > 12: 
        hour_12 -= 12
        period = "PM"
        
    label = f"{hour_12}:{minute_int:02d} {period}"

    if day_offset == 1:
        label += " (Next Day)"
    elif day_offset > 1:
        label += f" (+{day_offset} Days)"
    return label

def main():
    st.set_page_config(layout="wide") 
    st.title("Zonar Stop Time Analysis (Optimized)")

    st.sidebar.header("Analysis Configuration")
    op_day_cutoff_hour_config = st.sidebar.number_input(
        "Operational Day Cutoff Hour (e.g., 3 for 3 AM)",
        min_value=0, max_value=23, value=3, step=1, 
        help="Events before this hour on a calendar day are assigned to the previous operational day."
    )
    google_maps_api_key = st.sidebar.text_input("Google Maps API Key (Optional)", value="", type="password")

    default_session_state = {
        "filtered_df": pd.DataFrame(), "stops_df": pd.DataFrame(), "path_df": pd.DataFrame(),
        "origin_stop_name_id": None, "destination_stop_name_id": None, 
        "r_feet_origin": 200, "r_feet_dest": 200,
        "start_date_filter": None, "end_date_filter": None
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    c1_file, c2_file = st.columns(2)
    stops_file = c1_file.file_uploader("Upload Stops Classification CSV (Stop Name, Lat, Lon, Stop ID)", type=["csv"])
    path_file = c2_file.file_uploader("Upload Zonar Data CSV (Asset No., Date, Time, Dist, Lat, Lon)", type=["csv"])

    if stops_file and st.session_state.stops_df.empty: 
        try:
            stops_dtypes = {"Stop Name": "str", "Lat": "float64", "Lon": "float64", "Stop ID": "str"}
            stops_df_loaded = pd.read_csv(stops_file, dtype=stops_dtypes, usecols=list(stops_dtypes.keys()))
            
            required_stop_cols = ["Stop Name", "Lat", "Lon", "Stop ID"]
            if not set(required_stop_cols).issubset(stops_df_loaded.columns):
                st.error(f"Stops CSV must have columns: {', '.join(required_stop_cols)}")
            else:
                stops_df_loaded['_sort_stop_id_'] = pd.to_numeric(stops_df_loaded['Stop ID'], errors='coerce')
                stops_df_loaded.sort_values(by=['_sort_stop_id_', 'Stop Name'], inplace=True, na_position='last')
                stops_df_loaded.drop(columns=['_sort_stop_id_'], inplace=True)
                stops_df_loaded.reset_index(drop=True, inplace=True)
                stops_df_loaded["Stop_Name_ID"] = stops_df_loaded["Stop Name"] + " (" + stops_df_loaded["Stop ID"].astype(str) + ")"
                st.session_state["stops_df"] = stops_df_loaded
                st.success(f"Loaded {len(stops_df_loaded)} stops.")
        except Exception as e:
            st.error(f"Error loading stops CSV: {e}")
            st.session_state["stops_df"] = pd.DataFrame() 

    if path_file and st.session_state.path_df.empty: 
        try:
            path_req_cols = ["Asset No.", "Date", "Time(EST)", "Time(EDT)", "Distance Traveled(Miles)", "Lat", "Lon"]
            path_df_loaded = pd.read_csv(path_file, usecols=lambda c: c in path_req_cols + ["Time(EST)", "Time(EDT)"]) 

            essential_path_cols = ["Asset No.", "Date", "Distance Traveled(Miles)", "Lat", "Lon"]
            if not set(essential_path_cols).issubset(path_df_loaded.columns):
                st.error(f"Path CSV missing one or more mandatory columns: {', '.join(essential_path_cols)}")
                raise ValueError("Missing essential columns in path data.")

            time_col_to_use = "Time(EDT)" if "Time(EDT)" in path_df_loaded.columns and path_df_loaded["Time(EDT)"].notna().any() else "Time(EST)"
            if time_col_to_use not in path_df_loaded.columns:
                 st.error("Path CSV must contain 'Time(EST)' or 'Time(EDT)' column.")
                 raise ValueError("Missing time column in path data.")

            path_df_loaded["DateTime"] = pd.to_datetime(
                path_df_loaded["Date"] + " " + path_df_loaded[time_col_to_use],
                errors="coerce" 
            )
            path_df_loaded.dropna(subset=["DateTime"], inplace=True)

            if path_df_loaded.empty:
                st.warning("No valid DateTime entries found in Path data after parsing.")
            else:
                path_df_loaded["Asset No."] = path_df_loaded["Asset No."].astype("category")
                path_df_loaded["Lat"] = path_df_loaded["Lat"].astype("float64")
                path_df_loaded["Lon"] = path_df_loaded["Lon"].astype("float64")
                path_df_loaded["Distance Traveled(Miles)"] = path_df_loaded["Distance Traveled(Miles)"].astype("float32")
                
                # NOTE: Sorting is now done inside the analysis function, so it's not strictly needed here anymore,
                # but can be kept for consistency if desired.
                st.session_state["raw_path_df"] = path_df_loaded[
                    ["Asset No.", "DateTime", "Distance Traveled(Miles)", "Lat", "Lon"]
                ]
                st.success(f"Loaded {len(st.session_state['raw_path_df'])} path records.")
        except Exception as e:
            st.error(f"Error loading path CSV: {e}")
            st.session_state["path_df"] = pd.DataFrame() 
            st.session_state["raw_path_df"] = pd.DataFrame()


    if not st.session_state.get("raw_path_df", pd.DataFrame()).empty:
        path_df_for_filtering = st.session_state["raw_path_df"]
        min_dt_available = path_df_for_filtering["DateTime"].min().date()
        max_dt_available = path_df_for_filtering["DateTime"].max().date()

        if st.session_state.start_date_filter is None or not isinstance(st.session_state.start_date_filter, date):
            st.session_state.start_date_filter = min_dt_available
        if st.session_state.end_date_filter is None or not isinstance(st.session_state.end_date_filter, date):
            st.session_state.end_date_filter = max_dt_available
        
        current_start_date = max(min(st.session_state.start_date_filter, max_dt_available), min_dt_available)
        current_end_date = max(min(st.session_state.end_date_filter, max_dt_available), min_dt_available)
        if current_start_date > current_end_date: 
            current_start_date, current_end_date = current_end_date, current_start_date


        st.sidebar.markdown("**Filter Zonar Data by Date Range:**")
        s_filt = st.sidebar.date_input("Start date", value=current_start_date, min_value=min_dt_available, max_value=max_dt_available)
        e_filt = st.sidebar.date_input("End date", value=current_end_date, min_value=min_dt_available, max_value=max_dt_available)

        if s_filt and e_filt:
            st.session_state.start_date_filter, st.session_state.end_date_filter = s_filt, e_filt
            if s_filt <= e_filt:
                start_datetime = datetime.combine(s_filt, time.min)
                end_datetime = datetime.combine(e_filt, time.max)
                st.session_state["path_df"] = path_df_for_filtering[
                    (path_df_for_filtering["DateTime"] >= start_datetime) &
                    (path_df_for_filtering["DateTime"] <= end_datetime)
                ].copy() 
                if st.session_state["path_df"].empty:
                    st.sidebar.warning("Path data is empty after date filtering.")
                else:
                    st.sidebar.success(f"{len(st.session_state['path_df'])} records after date filter.")
            else:
                st.sidebar.error("Start date cannot be after end date. No date filter applied.")
                st.session_state["path_df"] = path_df_for_filtering.copy() 
    else:
         st.session_state["path_df"] = pd.DataFrame() 


    stops_ui = st.session_state.stops_df
    path_analysis_df = st.session_state.path_df

    if not stops_ui.empty and not path_analysis_df.empty:
        st.header("Stop Analysis Setup")
        col1, col2 = st.columns(2)
        
        stop_name_id_list = stops_ui["Stop_Name_ID"].unique().tolist()
        
        default_orig_idx = 0
        if st.session_state.origin_stop_name_id and st.session_state.origin_stop_name_id in stop_name_id_list:
            default_orig_idx = stop_name_id_list.index(st.session_state.origin_stop_name_id)
        
        default_dest_idx = 1 if len(stop_name_id_list) > 1 else 0
        if st.session_state.destination_stop_name_id and st.session_state.destination_stop_name_id in stop_name_id_list:
             default_dest_idx = stop_name_id_list.index(st.session_state.destination_stop_name_id)
        if default_orig_idx == default_dest_idx and len(stop_name_id_list) > 1 : 
            default_dest_idx = (default_orig_idx + 1) % len(stop_name_id_list)


        orig_choice_name_id = col1.selectbox("Origin Stop:", stop_name_id_list, index=default_orig_idx, key="sb_origin")
        dest_choice_name_id = col2.selectbox("Destination Stop:", stop_name_id_list, index=default_dest_idx, key="sb_destination")

        st.session_state.origin_stop_name_id = orig_choice_name_id
        st.session_state.destination_stop_name_id = dest_choice_name_id
        
        actual_origin_stop_name = stops_ui.loc[stops_ui["Stop_Name_ID"] == orig_choice_name_id, "Stop Name"].iloc[0]
        actual_dest_stop_name = stops_ui.loc[stops_ui["Stop_Name_ID"] == dest_choice_name_id, "Stop Name"].iloc[0]

        r_orig_val = col1.number_input("Origin Radius (ft)", min_value=1, value=st.session_state.r_feet_origin, step=10, key="num_r_orig")
        r_dest_val = col2.number_input("Destination Radius (ft)", min_value=1, value=st.session_state.r_feet_dest, step=10, key="num_r_dest")
        st.session_state.r_feet_origin, st.session_state.r_feet_dest = r_orig_val, r_dest_val


        st.sidebar.header("Filtering Options")
        max_dist_filter = st.sidebar.number_input("Max Trip Distance (miles, 0=auto cluster)", min_value=0.0, value=0.0, format="%.2f", step=0.5)
        gap_dist_filter = st.sidebar.number_input("Distance Cluster Gap (miles)", min_value=0.01, value=0.3, format="%.2f", step=0.05)
        dwell_gap_filter = st.sidebar.number_input("Dwell Time Cluster Gap (mins)", min_value=0.01, value=0.3, format="%.2f", step=0.05)
        iqr_mult_filter = st.sidebar.number_input("IQR Dwell Multiplier", min_value=0.1, value=1.5, format="%.1f", step=0.1)
        rm_tt_15_filter = st.sidebar.checkbox("Remove Top 15% Travel Time Outliers?", value=False)

        if st.button("🚀 Analyze Stop Crossings", type="primary", use_container_width=True):
            if actual_origin_stop_name == actual_dest_stop_name:
                st.error("Origin and Destination stops cannot be the same.")
            else:
                with st.spinner("Analyzing stop crossings with optimized hash table method..."):
                    # *** THIS IS THE ONLY LINE THAT CHANGES in main() ***
                    # We call the new, optimized function instead of the old one.
                    res_df = analyze_stop_crossings_hashtable(
                        stops_ui, path_analysis_df,
                        actual_origin_stop_name, actual_dest_stop_name,
                        r_orig_val, r_dest_val,
                        op_day_cutoff_hour=op_day_cutoff_hour_config
                    )
                
                if res_df.empty:
                    st.warning("No initial trips found between the selected stops with the given radii.")
                    st.session_state["filtered_df"] = pd.DataFrame()
                    return 

                final_df = res_df.copy()
                st.write(f"Initial trips found: {len(final_df)}")

                if max_dist_filter > 0:
                    final_df = final_df[final_df["Actual Distance (miles)"] <= max_dist_filter]
                else:
                    final_df = filter_by_gap_clustering_largest(final_df, gap_dist_filter)
                st.write(f"After distance filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "distance filter"): return

                final_df = filter_idle_by_gap_clustering_largest(final_df, "Origin Stop Idle (mins)", dwell_gap_filter)
                st.write(f"After origin dwell cluster filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "origin dwell cluster"): return
                
                final_df = filter_idle_by_gap_clustering_largest(final_df, "Destination Stop Idle (mins)", dwell_gap_filter)
                st.write(f"After destination dwell cluster filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "destination dwell cluster"): return

                final_df = filter_idle_by_iqr(final_df, "Origin Stop Idle (mins)", iqr_mult_filter)
                st.write(f"After origin dwell IQR filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "origin dwell IQR"): return

                final_df = filter_idle_by_iqr(final_df, "Destination Stop Idle (mins)", iqr_mult_filter)
                st.write(f"After destination dwell IQR filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "destination dwell IQR"): return

                if rm_tt_15_filter and not final_df.empty and "Travel Time" in final_df.columns and final_df["Travel Time"].dropna().shape[0] > 0:
                    final_df = final_df[final_df["Travel Time"] <= final_df["Travel Time"].quantile(0.85)]
                st.write(f"After travel time percentile filter: {len(final_df)}")
                if _check_empty_and_stop(final_df, "travel time percentile"): return

                if "Origin Stop Departure" in final_df.columns:
                    final_df["O_Depart_DT"] = pd.to_datetime(final_df["Origin Stop Departure"], errors='coerce')
                else:
                    final_df["O_Depart_DT"] = pd.NaT
                
                if "Travel Date" in final_df.columns:
                    final_df["Travel Date_dt"] = pd.to_datetime(final_df["Travel Date"], format="%m/%d/%Y", errors='coerce').dt.date
                else:
                    final_df["Travel Date_dt"] = final_df["O_Depart_DT"].apply(
                        lambda x: (x.date() - timedelta(days=1)) if pd.notnull(x) and x.hour < op_day_cutoff_hour_config else (x.date() if pd.notnull(x) else pd.NaT)
                    )

                final_df["Time of Day Numeric"] = calculate_time_of_day_adjusted_vectorized(
                    final_df["O_Depart_DT"], final_df["Travel Date_dt"]
                )
                final_df["Time of Day Label"] = final_df["Time of Day Numeric"].apply(format_time_of_day_label)
                
                st.session_state["filtered_df"] = final_df.reset_index(drop=True)
    
    elif (stops_file or not st.session_state.stops_df.empty) and \
         (path_file or not st.session_state.get("raw_path_df", pd.DataFrame()).empty) and \
         (st.session_state.path_df.empty):
        st.info("Path data is currently empty, likely due to date filtering. Adjust date range or upload new data.")
    elif not (stops_file or path_file):
        st.info("Please upload Stops and Zonar Path data CSV files to begin analysis.")


    display_df_final = st.session_state.get("filtered_df", pd.DataFrame())
    if not display_df_final.empty:
        st.header(f"📊 Analysis Results: {len(display_df_final)} Trips")
        
        disp_copy = display_df_final.copy()
        dt_display_cols = [
            "Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime",
            "Destination Stop Entry", "Destination Stop Departure"
        ]
        for c in dt_display_cols:
            if c in disp_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(disp_copy[c]):
                    disp_copy[c] = disp_copy[c].dt.strftime("%m/%d/%Y %I:%M:%S %p").fillna("N/A")
                else: 
                    disp_copy[c] = pd.to_datetime(disp_copy[c], errors='coerce').dt.strftime("%m/%d/%Y %I:%M:%S %p").fillna("N/A")
        
        cols_to_drop_for_display = ["O_Depart_DT", "Travel Date_dt", "Time of Day Numeric"] 
        st.dataframe(disp_copy.drop(columns=cols_to_drop_for_display, errors='ignore'), use_container_width=True)
    
        st.subheader("Summary Metrics")
        metrics_data = {
            "Metric": ["Total Trips", "Distinct Vehicles", "Avg Origin Dwell (mins)", 
                       "Avg Dest Dwell (mins)", "Avg Distance (miles)", "Avg Travel Time (mins)"],
            "Value": [
                len(display_df_final),
                display_df_final["Vehicle no."].nunique(),
                _safe_round(display_df_final["Origin Stop Idle (mins)"].mean()),
                _safe_round(display_df_final["Destination Stop Idle (mins)"].mean()),
                _safe_round(display_df_final["Actual Distance (miles)"].mean(), 3),
                _safe_round(display_df_final["Travel Time"].mean())
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        st.subheader("Distribution Plots")
        plot_df_secs = display_df_final.copy()
        for col_min, col_sec in [
            ("Travel Time", "Travel Time (secs)"),
            ("Origin Stop Idle (mins)", "Origin Stop Idle (secs)"),
            ("Destination Stop Idle (mins)", "Destination Stop Idle (secs)")
        ]:
            if col_min in plot_df_secs.columns:
                plot_df_secs[col_sec] = plot_df_secs[col_min].astype(float) * 60 

        _plot_dist(plot_df_secs, "Travel Time (secs)", "Travel Time (seconds)", "Travel Time")
        _plot_dist(plot_df_secs, "Origin Stop Idle (secs)", "Origin Dwell Time (seconds)", "Origin Dwell")
        _plot_dist(plot_df_secs, "Destination Stop Idle (secs)", "Destination Dwell Time (seconds)", "Destination Dwell")

        st.subheader("Time of Day Scatter Plots")
        scatter_df_tod = display_df_final.dropna(subset=["Time of Day Numeric", "O_Depart_DT"]).copy()
        
        if not scatter_df_tod.empty:
            min_tod_plot = scatter_df_tod["Time of Day Numeric"].min() -1 
            max_tod_plot = scatter_df_tod["Time of Day Numeric"].max() + 1
            min_tod_plot = max(0, min_tod_plot) 
            
            if max_tod_plot - min_tod_plot < 24:
                max_tod_plot = min_tod_plot + 24
            
            x_axis_tod = alt.X("Time of Day Numeric:Q", title="Time of Day (Origin Departure)",
                               scale=alt.Scale(domain=[min_tod_plot, max_tod_plot], clamp=True),
                               axis=alt.Axis(labelAngle=-45, tickCount=12)) 

            _plot_scatter_tod(scatter_df_tod, "Travel Time", "Travel Time (mins)", x_axis_tod)
            _plot_scatter_tod(scatter_df_tod, "Origin Stop Idle (mins)", "Origin Dwell (mins)", x_axis_tod)
            _plot_scatter_tod(scatter_df_tod, "Destination Stop Idle (mins)", "Dest. Dwell (mins)", x_axis_tod)
        else:
            st.write("Not enough data for Time of Day scatter plots (after removing entries with missing time data).")

        st.subheader("Snap-to-Roads Map (Single Trip)")
        if not google_maps_api_key:
            st.warning("Enter a Google Maps API Key in the sidebar to enable the Snap-to-Roads map feature.")
        elif not display_df_final.empty:
            map_options_list = []
            map_df_selection = display_df_final.copy()
            map_df_selection["Last Ping In Origin DateTime_map"] = pd.to_datetime(
                map_df_selection["Last Ping In Origin DateTime"], errors='coerce'
            )
            map_df_selection["D_Arrival_DT_map"] = pd.to_datetime(map_df_selection["Destination Stop Entry"], errors='coerce')

            for idx, row in map_df_selection.head(100).iterrows():
                label = f"Trip (Idx {row.name}) | Bus: {row['Vehicle no.']} | Travel Time: {row['Travel Time']:.1f}m"
                map_options_list.append((label, row.name)) 

            if map_options_list:
                choice_label = st.selectbox("Select a trip to display on map (shows first 100 filtered trips):",
                                            [o[0] for o in map_options_list])
                if choice_label:
                    chosen_original_idx = next(o[1] for o in map_options_list if o[0] == choice_label)
                    chosen_row = map_df_selection.loc[chosen_original_idx]
                    
                    ping_start_time_for_map = chosen_row.get("Last Ping In Origin DateTime_map", pd.NaT)
                    d_arrival_dt_map = chosen_row.get("D_Arrival_DT_map", pd.NaT)

                    if pd.isna(ping_start_time_for_map) or pd.isna(d_arrival_dt_map):
                        st.warning(f"Selected trip (Index {chosen_original_idx}) is missing necessary timestamps (last ping in origin or destination entry) for map display.")
                    else:
                        path_for_map_df = st.session_state.get("path_df", pd.DataFrame()) 
                        stops_for_map_df = st.session_state.get("stops_df", pd.DataFrame())

                        if path_for_map_df.empty or stops_for_map_df.empty:
                            st.warning("Path or Stops data is not available for map display.")
                        else:
                            bus_trip_pings = path_for_map_df[
                                (path_for_map_df["Asset No."] == chosen_row["Vehicle no."]) &
                                (path_for_map_df["DateTime"] >= ping_start_time_for_map) &
                                (path_for_map_df["DateTime"] <= d_arrival_dt_map) 
                            ].copy()
                            
                            coords_original = [
                                (r["Lat"], r["Lon"]) for _, r in bus_trip_pings.sort_values("DateTime").iterrows()
                                if pd.notna(r["Lat"]) and pd.notna(r["Lon"])
                            ]

                            if not coords_original:
                                st.warning("No valid GPS coordinates found for the selected trip segment.")
                            else:
                                st.write(f"Displaying map for Trip Index {chosen_original_idx}. Found {len(coords_original)} raw GPS points.")
                                with st.spinner("Snapping coordinates to roads..."):
                                     snapped_coords = snap_to_roads(coords_original, google_maps_api_key)
                                
                                origin_stop_info = stops_for_map_df[stops_for_map_df["Stop Name"] == actual_origin_stop_name].iloc[0]
                                dest_stop_info = stops_for_map_df[stops_for_map_df["Stop Name"] == actual_dest_stop_name].iloc[0]
                                
                                embed_snapped_polyline_map(
                                    coords_original, snapped_coords,
                                    origin_stop_info["Lat"], origin_stop_info["Lon"],
                                    dest_stop_info["Lat"], dest_stop_info["Lon"],
                                    st.session_state.r_feet_origin, st.session_state.r_feet_dest,
                                    google_maps_api_key
                                )
            else:
                st.write("No trips available in the filtered data for map selection.")
        elif google_maps_api_key: 
             st.write("No filtered trip data available to display on the map.")


def _check_empty_and_stop(df: pd.DataFrame, filter_name: str) -> bool:
    """Checks if DataFrame is empty after a filter, warns, clears session state, and stops if so."""
    if df.empty:
        st.warning(f"All trips removed by {filter_name} filter.")
        st.session_state["filtered_df"] = pd.DataFrame() 
        st.stop() 
        return True
    return False

def _safe_round(val, decimals=2):
    if isinstance(val, (int, float, np.number)): 
        if pd.notnull(val): 
            return round(float(val), decimals)
        else: 
            return "N/A" 
    elif pd.isna(val): 
            return "N/A"
    return "N/A" 


def _plot_dist(df: pd.DataFrame, col_name: str, title_x: str, base_title: str):
    if col_name in df.columns:
        data_to_plot = df[[col_name]].dropna()
        if not data_to_plot.empty:
            st.write(f"#### Distribution of {base_title}")
            try:
                chart = alt.Chart(data_to_plot).mark_bar().encode(
                    alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=30), title=title_x, type="quantitative"),
                    alt.Y('count()', title="Number of Trips", type="quantitative")
                ).properties(height=300, title=f"{base_title} Distribution")
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating distribution plot for {base_title}: {e}")
        else:
            st.write(f"Not enough valid data for {base_title} distribution plot.")
    else:
        st.write(f"Column '{col_name}' not found for {base_title} distribution plot.")


def _plot_scatter_tod(df: pd.DataFrame, y_col: str, y_title: str, x_axis: alt.X):
    required_cols = ["Time of Day Numeric", y_col, "O_Depart_DT", "Vehicle no.", "Time of Day Label"]
    if all(col in df.columns for col in required_cols):
        subset_df_scatter = df.copy()
        if not pd.api.types.is_numeric_dtype(subset_df_scatter[y_col]):
            subset_df_scatter[y_col] = pd.to_numeric(subset_df_scatter[y_col], errors='coerce')
        
        subset_df_scatter.dropna(subset=required_cols, inplace=True) 
        
        if not subset_df_scatter.empty:
            st.write(f"#### {y_title} vs. Time of Day")

            if len(subset_df_scatter["Time of Day Numeric"].unique()) >= 2 and len(subset_df_scatter[y_col].unique()) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(subset_df_scatter["Time of Day Numeric"], subset_df_scatter[y_col])
                    st.markdown("**Regression Statistics:**")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric(label="Slope", value=f"{slope:.4f}")
                    stat_col2.metric(label="Intercept", value=f"{intercept:.4f}")
                    stat_col3.metric(label="R-squared (R²)", value=f"{r_value**2:.4f}")
                except Exception as e:
                    st.warning(f"Could not calculate regression statistics for {y_title}: {e}")

            point_tooltip_items = [
                alt.Tooltip("Vehicle no.:N", title="Vehicle No."),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.2f"),
                alt.Tooltip("Time of Day Label:N", title="Time of Day (Adjusted)"),
                alt.Tooltip("O_Depart_DT:T", title="Departure Timestamp", format='%m/%d/%Y %I:%M:%S %p')
            ]
            for idle_col_name, idle_title_tooltip in [
                ("Origin Stop Idle (mins)", "Origin Idle (mins)"),
                ("Destination Stop Idle (mins)", "Dest. Idle (mins)")]:
                if idle_col_name in subset_df_scatter.columns and y_col != idle_col_name:
                    if pd.api.types.is_numeric_dtype(subset_df_scatter[idle_col_name]):
                        point_tooltip_items.append(alt.Tooltip(f"{idle_col_name}:Q", title=idle_title_tooltip, format=",.2f"))

            try:
                base = alt.Chart(subset_df_scatter).encode(
                    x=x_axis, 
                    y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(zero=False)) 
                )
                points = base.mark_circle(size=60, opacity=0.7).encode(
                    tooltip=point_tooltip_items,
                    color=alt.Color("Vehicle no.:N", legend=None) 
                ).interactive() 
                line = base.transform_regression(on="Time of Day Numeric", regression=y_col).mark_line(color="red")          
                chart = (points + line).properties(height=400, title=f"Chart: {y_title} vs. Origin Departure Time")
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                 st.error(f"Error generating scatter plot for {y_title}: {e}")
        else:
            st.write(f"Not enough valid data for '{y_title} vs. Time of Day' scatter plot (after dropping NaNs).")
    else:
        missing = [col for col in required_cols if col not in df.columns]
        st.write(f"Required column(s) missing for scatter plot '{y_title}': {', '.join(missing)}")


if __name__ == "__main__":
    # Remove the original functions that are now replaced to avoid clutter
    # if 'analyze_stop_crossings' in globals(): del globals()['analyze_stop_crossings']
    # if '_process_single_bus' in globals(): del globals()['_process_single_bus']
    main()