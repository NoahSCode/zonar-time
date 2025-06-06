import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import altair as alt
import requests
import streamlit.components.v1 as components
from scipy.stats import linregress
from dataclasses import dataclass

################################################################################
# 0) Data Class for State Management
################################################################################
@dataclass
class BusState:
    """Holds the state of a single bus during chronological processing."""
    inside_origin: bool = False
    inside_dest: bool = False
    origin_arrival_time: pd.Timestamp = pd.NaT
    origin_depart_time: pd.Timestamp = pd.NaT
    dest_arrival_time: pd.Timestamp = pd.NaT
    prev_datetime: pd.Timestamp = pd.NaT
    prev_in_origin_zone: bool = False
    prev_in_dest_zone: bool = False
    prev_dist_traveled: float = np.nan
    origin_depart_start_dist: float = np.nan
    last_ping_in_origin_dt_current_trip: pd.Timestamp = pd.NaT

################################################################################
# 1) Utility: Haversine & Interpolation
################################################################################
EARTH_RADIUS_FEET = 20902231

def haversine_distance_vectorized(lat1_arr: np.ndarray, lon1_arr: np.ndarray, lat2_scalar: float, lon2_scalar: float) -> np.ndarray:
    lat1_rad, lon1_rad = np.radians(lat1_arr), np.radians(lon1_arr)
    lat2_rad, lon2_rad = np.radians(lat2_scalar), np.radians(lon2_scalar)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_FEET * c

def interpolate_time(t1: pd.Timestamp, t2: pd.Timestamp) -> pd.Timestamp | None:
    if pd.isna(t1) or pd.isna(t2): return None
    t1, t2 = pd.Timestamp(t1), pd.Timestamp(t2)
    return (t1 + (t2 - t1) / 2).replace(microsecond=0)

################################################################################
# 2) Optimized Core Analysis Function
################################################################################
def analyze_stop_crossings_hashtable(
    stops_df: pd.DataFrame, path_df: pd.DataFrame, origin_stop: str, destination_stop: str,
    radius_feet_origin: float, radius_feet_destination: float, op_day_cutoff_hour: int
) -> pd.DataFrame:
    """Analyzes stop crossings using a single chronological pass and a hash table."""
    if stops_df.empty or path_df.empty: return pd.DataFrame()

    try:
        origin_row = stops_df[stops_df["Stop Name"] == origin_stop].iloc[0]
        dest_row = stops_df[stops_df["Stop Name"] == destination_stop].iloc[0]
    except IndexError:
        st.error(f"Origin or Destination not found in stops data.")
        return pd.DataFrame()

    # Create a local copy for modification to avoid side effects
    df = path_df.copy()

    # Standardize column names for predictable access
    rename_map = {"Asset No.": "AssetNo", "Distance Traveled(Miles)": "DistanceTraveledMiles"}
    df.rename(columns=rename_map, inplace=True, errors='ignore')

    df.sort_values("DateTime", inplace=True)
    df['in_origin_zone'] = haversine_distance_vectorized(df["Lat"].values, df["Lon"].values, origin_row["Lat"], origin_row["Lon"]) <= radius_feet_origin
    df['in_dest_zone'] = haversine_distance_vectorized(df["Lat"].values, df["Lon"].values, dest_row["Lat"], dest_row["Lon"]) <= radius_feet_destination

    bus_states, all_results_list = {}, []

    for ping in df.itertuples(index=False):
        bus_id, curr_datetime = ping.AssetNo, ping.DateTime
        curr_in_origin, curr_in_dest = ping.in_origin_zone, ping.in_dest_zone
        curr_dist_traveled = ping.DistanceTraveledMiles

        state = bus_states.get(bus_id, BusState())
        bus_states[bus_id] = state

        if not state.inside_origin and curr_in_origin:
            state.inside_origin = True
            state.origin_arrival_time = curr_datetime
        
        if state.inside_origin and state.prev_in_origin_zone and not curr_in_origin:
            state.inside_origin = False
            state.origin_depart_time = interpolate_time(state.prev_datetime, curr_datetime)
            state.origin_depart_start_dist = state.prev_dist_traveled
            state.last_ping_in_origin_dt_current_trip = state.prev_datetime

        if not pd.isna(state.origin_depart_time) and not state.inside_dest and curr_in_dest:
            if curr_datetime < state.origin_depart_time:
                state.prev_datetime, state.prev_in_origin_zone, state.prev_in_dest_zone, state.prev_dist_traveled = curr_datetime, curr_in_origin, curr_in_dest, curr_dist_traveled
                continue

            state.inside_dest = True
            state.dest_arrival_time = curr_datetime
            
            actual_distance = curr_dist_traveled - state.origin_depart_start_dist if pd.notna(state.origin_depart_start_dist) and pd.notna(curr_dist_traveled) else 0.0
            origin_idle_mins = (state.origin_depart_time - state.origin_arrival_time).total_seconds() / 60.0
            travel_time_mins = (state.dest_arrival_time - state.origin_depart_time).total_seconds() / 60.0

            travel_date_str = None
            if pd.notnull(state.origin_depart_time):
                event_dt = state.origin_depart_time
                travel_date_to_format = event_dt.date() - timedelta(days=1) if event_dt.hour < op_day_cutoff_hour else event_dt.date()
                travel_date_str = travel_date_to_format.strftime("%m/%d/%Y")

            all_results_list.append({
                "Vehicle no.": bus_id, "Travel Date": travel_date_str,
                "Origin Stop": origin_stop, "Destination Stop": destination_stop,
                "Origin Stop Arrival": state.origin_arrival_time, "Origin Stop Departure": state.origin_depart_time,
                "Last Ping In Origin DateTime": state.last_ping_in_origin_dt_current_trip,
                "Origin Stop Idle (mins)": round(origin_idle_mins, 2),
                "Destination Stop Entry": state.dest_arrival_time, "Destination Stop Departure": pd.NaT,
                "Destination Stop Idle (mins)": None,
                "Actual Distance (miles)": round(actual_distance, 3) if actual_distance >= 0 else 0,
                "Travel Time": round(travel_time_mins, 2) if travel_time_mins >=0 else 0
            })
            
            state.origin_arrival_time, state.origin_depart_time, state.origin_depart_start_dist, state.last_ping_in_origin_dt_current_trip = pd.NaT, pd.NaT, np.nan, pd.NaT

        if state.inside_dest and state.prev_in_dest_zone and not curr_in_dest:
            dest_depart_time_val = interpolate_time(state.prev_datetime, curr_datetime)
            for i in range(len(all_results_list) - 1, -1, -1):
                record = all_results_list[i]
                if (record["Vehicle no."] == bus_id and pd.isna(record["Destination Stop Departure"]) and record["Destination Stop Entry"] == state.dest_arrival_time):
                    record["Destination Stop Departure"] = dest_depart_time_val
                    if pd.notna(dest_depart_time_val) and pd.notna(record["Destination Stop Entry"]):
                        dest_idle_mins = (dest_depart_time_val - record["Destination Stop Entry"]).total_seconds() / 60.0
                        record["Destination Stop Idle (mins)"] = round(dest_idle_mins, 2)
                    break
            bus_states[bus_id] = BusState()

        state.prev_datetime, state.prev_in_origin_zone, state.prev_in_dest_zone, state.prev_dist_traveled = curr_datetime, curr_in_origin, curr_in_dest, curr_dist_traveled

    if not all_results_list: return pd.DataFrame()

    results_df = pd.DataFrame(all_results_list)
    dt_cols = ["Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime", "Destination Stop Entry", "Destination Stop Departure"]
    for col in dt_cols: results_df[col] = pd.to_datetime(results_df[col], errors='coerce')
    
    final_cols = ["Vehicle no.", "Travel Date", "Origin Stop", "Destination Stop", "Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime", "Origin Stop Idle (mins)", "Destination Stop Entry", "Destination Stop Departure", "Destination Stop Idle (mins)", "Actual Distance (miles)", "Travel Time"]
    return results_df[[col for col in final_cols if col in results_df.columns]]

################################################################################
# 3) Filtering, Plotting & Other Utilities
################################################################################
def filter_by_gap_clustering_largest(df: pd.DataFrame, gap_threshold: float) -> pd.DataFrame:
    if df.empty or "Actual Distance (miles)" not in df.columns: return df.copy()
    sorted_df = df.dropna(subset=["Actual Distance (miles)"]).sort_values(by="Actual Distance (miles)")
    if sorted_df.empty or len(sorted_df) < 2: return df.copy()
    distances, clusters_indices, current_cluster = sorted_df["Actual Distance (miles)"].values, [], [sorted_df.index[0]]
    if len(distances) == 1: clusters_indices.append(current_cluster)
    else:
        for i in range(len(distances) - 1):
            if (distances[i+1] - distances[i]) >= gap_threshold: clusters_indices.append(current_cluster); current_cluster = [sorted_df.index[i+1]]
            else: current_cluster.append(sorted_df.index[i+1])
        if current_cluster: clusters_indices.append(current_cluster)
    if not clusters_indices: return pd.DataFrame(columns=df.columns)
    return df.loc[max(clusters_indices, key=len)].copy()

def filter_idle_by_gap_clustering_largest(df: pd.DataFrame, idle_col: str, gap_threshold: float) -> pd.DataFrame:
    if df.empty or idle_col not in df.columns: return df.copy()
    sub_df = df.dropna(subset=[idle_col])
    if sub_df.empty or len(sub_df) < 2: return df.copy()
    sorted_sub = sub_df.sort_values(by=idle_col)
    idle_vals, clusters_indices, current_cluster = sorted_sub[idle_col].values, [], [sorted_sub.index[0]]
    if len(idle_vals) == 1: clusters_indices.append(current_cluster)
    else:
        for i in range(len(idle_vals) - 1):
            if (idle_vals[i+1] - idle_vals[i]) >= gap_threshold: clusters_indices.append(current_cluster); current_cluster = [sorted_sub.index[i+1]]
            else: current_cluster.append(sorted_sub.index[i+1])
        if current_cluster: clusters_indices.append(current_cluster)
    if not clusters_indices: return pd.DataFrame(columns=df.columns)
    return df.loc[max(clusters_indices, key=len)].copy()

def filter_idle_by_iqr(df: pd.DataFrame, idle_col: str, multiplier: float) -> pd.DataFrame:
    if df.empty or idle_col not in df.columns: return df.copy()
    sub = df.dropna(subset=[idle_col])
    if sub.empty: return df.copy()
    q1, q3 = sub[idle_col].quantile(0.25), sub[idle_col].quantile(0.75)
    iqr = q3 - q1
    upper_cut = q3 if iqr == 0 else q3 + multiplier * iqr
    return df[(df[idle_col] <= upper_cut) | df[idle_col].isna()].copy()

def snap_to_roads(coords: list[tuple[float, float]], api_key: str) -> list:
    if not coords: return []
    snapped_points, BATCH_SIZE = [], 100
    for i in range(0, len(coords), BATCH_SIZE):
        chunk = coords[i:i + BATCH_SIZE]
        path_param = "|".join(f"{c[0]},{c[1]}" for c in chunk)
        url = f"https://roads.googleapis.com/v1/snapToRoads?path={path_param}&interpolate=false&key={api_key}"
        try:
            resp = requests.get(url, timeout=10); resp.raise_for_status(); data = resp.json()
            for p in data.get("snappedPoints", []): snapped_points.append((p["location"]["latitude"], p["location"]["longitude"]))
        except requests.exceptions.RequestException as e: st.error(f"Snap to Roads API error: {e}"); return []
        except Exception as e: st.error(f"Error processing Snap to Roads response: {e}"); return []
    return snapped_points

def embed_snapped_polyline_map(original_coords, snapped_coords, origin_stop, dest_stop, radius_feet_origin, radius_feet_destination, api_key):
    if not original_coords and not snapped_coords:
        st.write("No coordinates to display on map."); return
    radius_m_orig, radius_m_dest = float(radius_feet_origin) * 0.3048, float(radius_feet_destination) * 0.3048
    center_lat, center_lon = (snapped_coords[0] if snapped_coords else original_coords[0]) if (snapped_coords or original_coords) else (origin_stop['Lat'], origin_stop['Lon'])
    snapped_js = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in snapped_coords)
    orig_js = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in original_coords)
    html_code = f"""<!DOCTYPE html><html><head><style>#map{{height:100%;width:100%}}html,body{{margin:0;padding:0;height:100%}}</style></head><body><div id="map"></div><script>
    function initMap(){{var map=new google.maps.Map(document.getElementById('map'),{{zoom:13,center:{{lat:{center_lat},lng:{center_lon}}}}});
    new google.maps.Circle({{strokeColor:"#0000FF",strokeOpacity:0.8,strokeWeight:2,fillColor:"#0000FF",fillOpacity:0.1,map:map,center:{{lat:{origin_stop['Lat']},lng:{origin_stop['Lon']}}},radius:{radius_m_orig}}});
    new google.maps.Marker({{position:{{lat:{origin_stop['Lat']},lng:{origin_stop['Lon']}}},map:map,title:"Origin"}});
    new google.maps.Circle({{strokeColor:"#00FF00",strokeOpacity:0.8,strokeWeight:2,fillColor:"#00FF00",fillOpacity:0.1,map:map,center:{{lat:{dest_stop['Lat']},lng:{dest_stop['Lon']}}},radius:{radius_m_dest}}});
    new google.maps.Marker({{position:{{lat:{dest_stop['Lat']},lng:{dest_stop['Lon']}}},map:map,title:"Destination"}});
    var snappedCoords=[{snapped_js}];if(snappedCoords.length>1){{new google.maps.Polyline({{path:snappedCoords,geodesic:true,strokeColor:"#4285F4",strokeOpacity:1.0,strokeWeight:4}}).setMap(map);}}
    var originalCoords=[{orig_js}];originalCoords.forEach(function(c){{new google.maps.Marker({{position:c,map:map,icon:{{path:google.maps.SymbolPath.CIRCLE,scale:2,fillColor:'#FF0000',fillOpacity:0.7,strokeColor:'#FF0000',strokeWeight:1}}}});}});}}
    </script><script async src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap"></script></body></html>"""
    components.html(html_code, height=600)

def calculate_time_of_day_adjusted_vectorized(s_depart_dt: pd.Series, s_travel_date_dt: pd.Series) -> pd.Series:
    if s_depart_dt.empty or s_travel_date_dt.empty: return pd.Series(index=s_depart_dt.index, dtype=float)
    depart_dt = pd.to_datetime(s_depart_dt, errors='coerce')
    time_val = depart_dt.dt.hour + depart_dt.dt.minute / 60.0
    travel_date_dt_ts = pd.to_datetime(s_travel_date_dt); departure_date_only = depart_dt.dt.date
    one_day_after = (departure_date_only > s_travel_date_dt) & (departure_date_only <= (s_travel_date_dt + pd.to_timedelta(1, unit='D')))
    two_days_after = (departure_date_only > (s_travel_date_dt + pd.to_timedelta(1, unit='D')))
    time_val = np.where(one_day_after, time_val + 24.0, time_val); time_val = np.where(two_days_after, time_val + 48.0, time_val)
    time_val[depart_dt.isna() | travel_date_dt_ts.isna()] = np.nan
    return pd.Series(time_val, index=s_depart_dt.index)

def format_time_of_day_label(time_val_24hr: float) -> str:
    if pd.isna(time_val_24hr): return "N/A"
    day_offset = int(time_val_24hr // 24); hour_of_day_24 = time_val_24hr % 24
    hour_int, minute_int = int(hour_of_day_24), int(round((hour_of_day_24 - int(hour_of_day_24)) * 60))
    if minute_int == 60: hour_int += 1; minute_int = 0
    if hour_int == 24: hour_int = 0; day_offset += 1
    period, hour_12 = ("PM" if 12 <= hour_int < 24 else "AM"), (hour_int if hour_int in [0, 12] else hour_int % 12)
    if hour_12 == 0: hour_12 = 12
    label = f"{hour_12}:{minute_int:02d} {period}"
    if day_offset > 0: label += f" (+{day_offset} Day{'s' if day_offset > 1 else ''})"
    return label

def _check_empty_and_stop(df: pd.DataFrame, filter_name: str) -> bool:
    if df.empty: st.warning(f"All trips removed by {filter_name} filter."); st.session_state.filtered_df = pd.DataFrame(); st.stop()
    return False

def _safe_round(val, decimals=2):
    return round(float(val), decimals) if isinstance(val, (int, float, np.number)) and pd.notnull(val) else "N/A"

def _plot_dist(df: pd.DataFrame, col_name: str, title_x: str, base_title: str):
    if col_name in df.columns and not df[col_name].dropna().empty:
        st.write(f"#### Distribution of {base_title}")
        chart = alt.Chart(df).mark_bar().encode(alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=30), title=title_x), alt.Y('count()', title="Number of Trips")).properties(height=300, title=f"{base_title} Distribution")
        st.altair_chart(chart, use_container_width=True)

def _plot_scatter_tod(df: pd.DataFrame, y_col: str, y_title: str, x_axis: alt.X):
    req_cols = ["Time of Day Numeric", y_col, "O_Depart_DT", "Vehicle no.", "Time of Day Label"]
    if all(c in df.columns for c in req_cols):
        subset = df.dropna(subset=req_cols).copy()
        if not subset.empty:
            st.write(f"#### {y_title} vs. Time of Day")
            if len(x_vals := subset["Time of Day Numeric"].values) > 1 and len(y_vals := subset[y_col].values) > 1:
                try:
                    slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                    st.markdown("**Regression Statistics:**")
                    c1,c2,c3 = st.columns(3); c1.metric("Slope", f"{slope:.4f}"); c2.metric("Intercept", f"{intercept:.4f}"); c3.metric("R-squared (R²)", f"{r_value**2:.4f}")
                except Exception as e: st.warning(f"Could not calculate regression stats: {e}")
            base = alt.Chart(subset).encode(x=x_axis, y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(zero=False)))
            tooltip_items = [
                alt.Tooltip("Vehicle no.:N", title="Vehicle No."),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.2f"),
                alt.Tooltip("Time of Day Label:N", title="Time of Day"),
                alt.Tooltip("O_Depart_DT:T", title="Departure Time", format='%Y-%m-%d %H:%M')
            ]
            points = base.mark_circle(size=60, opacity=0.7).encode(tooltip=tooltip_items, color=alt.Color("Vehicle no.:N", legend=None)).interactive()
            line = base.transform_regression(on="Time of Day Numeric", regression=y_col).mark_line(color="red")
            st.altair_chart((points + line).properties(height=400, title=f"Chart: {y_title} vs. Time of Day"), use_container_width=True)

################################################################################
# 7) Main Application
################################################################################
def main():
    st.set_page_config(layout="wide")
    st.title("Zonar Stop Time Analysis")

    st.sidebar.header("Analysis Configuration")
    op_day_cutoff_hour_config = st.sidebar.number_input("Operational Day Cutoff Hour", 0, 23, 3, 1)
    google_maps_api_key = st.sidebar.text_input("Google Maps API Key (Optional)", type="password")

    for key in ["stops_df", "raw_path_df", "path_df", "filtered_df", "processed_stops_files", "processed_path_files", "start_date_filter", "end_date_filter"]:
        if key not in st.session_state: st.session_state[key] = None if "files" in key or "date" in key else pd.DataFrame()

    c1, c2 = st.columns(2)
    stops_files = c1.file_uploader("Upload Stops CSV(s)", type=["csv"], accept_multiple_files=True)
    path_files = c2.file_uploader("Upload Path Data CSV(s)", type=["csv"], accept_multiple_files=True)

    if stops_files and stops_files != st.session_state.processed_stops_files:
        try:
            dfs = [pd.read_csv(f, dtype={"Stop Name": "str", "Lat": "float64", "Lon": "float64", "Stop ID": "str"}, usecols=["Stop Name", "Lat", "Lon", "Stop ID"]) for f in stops_files]
            df = pd.concat(dfs, ignore_index=True).drop_duplicates()
            df['_sort_stop_id_'] = pd.to_numeric(df['Stop ID'], errors='coerce')
            df.sort_values(by=['_sort_stop_id_', 'Stop Name'], inplace=True, na_position='last')
            df["Stop_Name_ID"] = df["Stop Name"] + " (" + df["Stop ID"].astype(str) + ")"
            st.session_state.stops_df = df.drop(columns=['_sort_stop_id_']).reset_index(drop=True)
            st.session_state.processed_stops_files = stops_files
            st.success(f"Loaded {len(df)} unique stops from {len(stops_files)} file(s).")
        except Exception as e: st.error(f"Error loading stops CSV(s): {e}"); st.session_state.stops_df, st.session_state.processed_stops_files = pd.DataFrame(), None

    if path_files and path_files != st.session_state.processed_path_files:
        dfs = []
        try:
            with st.spinner(f"Loading data from {len(path_files)} file(s)..."):
                for f in path_files:
                    df = pd.read_csv(f)
                    rename_map = {"Asset No.": "AssetNo", "Distance Traveled(Miles)": "DistanceTraveledMiles", "Time(EST)": "TimeEST", "Time(EDT)": "TimeEDT"}
                    df.rename(columns=rename_map, inplace=True, errors='ignore')
                    if not set(["AssetNo", "Date", "DistanceTraveledMiles", "Lat", "Lon"]).issubset(df.columns): continue
                    time_col = "TimeEDT" if "TimeEDT" in df.columns and df["TimeEDT"].notna().any() else "TimeEST"
                    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df[time_col], errors="coerce")
                    df.dropna(subset=["DateTime"], inplace=True)
                    dfs.append(df)
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                for col, dtype in {"AssetNo": "category", "Lat": "float64", "Lon": "float64", "DistanceTraveledMiles": "float32"}.items(): combined_df[col] = combined_df[col].astype(dtype)
                st.session_state.raw_path_df = combined_df[["AssetNo", "DateTime", "DistanceTraveledMiles", "Lat", "Lon"]]
                st.session_state.processed_path_files = path_files
                st.session_state.path_df, st.session_state.filtered_df = pd.DataFrame(), pd.DataFrame()
                st.session_state.start_date_filter, st.session_state.end_date_filter = None, None
                st.success(f"Loaded {len(combined_df)} path records from {len(dfs)} valid file(s)."); st.rerun()
        except Exception as e: st.error(f"Error loading path CSV(s): {e}"); st.session_state.raw_path_df, st.session_state.processed_path_files = pd.DataFrame(), None

    if not st.session_state.raw_path_df.empty:
        df_filter = st.session_state.raw_path_df
        min_dt, max_dt = df_filter["DateTime"].min().date(), df_filter["DateTime"].max().date()
        start_date = st.session_state.start_date_filter or min_dt
        end_date = st.session_state.end_date_filter or max_dt
        st.sidebar.markdown("**Filter Data by Date Range:**")
        s_filt = st.sidebar.date_input("Start date", value=start_date, min_value=min_dt, max_value=max_dt)
        e_filt = st.sidebar.date_input("End date", value=end_date, min_value=min_dt, max_value=max_dt)
        if s_filt and e_filt:
            st.session_state.start_date_filter, st.session_state.end_date_filter = s_filt, e_filt
            if s_filt <= e_filt:
                st.session_state.path_df = df_filter[(df_filter["DateTime"] >= datetime.combine(s_filt, time.min)) & (df_filter["DateTime"] <= datetime.combine(e_filt, time.max))].copy()
                st.sidebar.success(f"{len(st.session_state.path_df)} records in range.")
            else: st.sidebar.error("Start date must be before end date."); st.session_state.path_df = pd.DataFrame()

    stops_ui, path_analysis_df = st.session_state.stops_df, st.session_state.path_df
    if not stops_ui.empty and not path_analysis_df.empty:
        st.header("Stop Analysis Setup")
        c1, c2 = st.columns(2)
        stop_list = stops_ui["Stop_Name_ID"].unique().tolist()
        orig_choice = c1.selectbox("Origin Stop:", stop_list, index=0)
        dest_choice = c2.selectbox("Destination Stop:", stop_list, index=min(1, len(stop_list)-1))
        origin_stop_info = stops_ui.loc[stops_ui["Stop_Name_ID"] == orig_choice].iloc[0]
        dest_stop_info = stops_ui.loc[stops_ui["Stop_Name_ID"] == dest_choice].iloc[0]
        r_orig = c1.number_input("Origin Radius (ft)", 1, value=200, step=10)
        r_dest = c2.number_input("Destination Radius (ft)", 1, value=200, step=10)
        
        st.sidebar.header("Filtering Options")
        max_dist = st.sidebar.number_input("Max Trip Distance (0=auto)", 0.0, value=0.0, format="%.2f")
        dist_gap = st.sidebar.number_input("Distance Cluster Gap (miles)", 0.01, value=0.3, format="%.2f")
        dwell_gap = st.sidebar.number_input("Dwell Time Cluster Gap (mins)", 0.01, value=0.3, format="%.2f")
        iqr_mult = st.sidebar.number_input("IQR Dwell Multiplier", 0.1, value=1.5, format="%.1f")
        rm_tt_outliers = st.sidebar.checkbox("Remove Top 15% Travel Time Outliers?", False)

        if st.button("🚀 Analyze Stop Crossings", type="primary", use_container_width=True):
            if origin_stop_info["Stop Name"] == dest_stop_info["Stop Name"]: st.error("Origin and Destination must be different.")
            else:
                with st.spinner("Analyzing..."):
                    df = analyze_stop_crossings_hashtable(stops_ui, path_analysis_df, origin_stop_info["Stop Name"], dest_stop_info["Stop Name"], r_orig, r_dest, op_day_cutoff_hour_config)
                if df.empty: st.warning("No trips found."); st.session_state.filtered_df = pd.DataFrame(); return

                st.write(f"Initial trips: {len(df)}")
                df = df[df["Actual Distance (miles)"] <= max_dist] if max_dist > 0 else filter_by_gap_clustering_largest(df, dist_gap)
                if _check_empty_and_stop(df, "distance"): return
                df = filter_idle_by_gap_clustering_largest(df, "Origin Stop Idle (mins)", dwell_gap); _check_empty_and_stop(df, "origin dwell cluster")
                df = filter_idle_by_gap_clustering_largest(df, "Destination Stop Idle (mins)", dwell_gap); _check_empty_and_stop(df, "destination dwell cluster")
                df = filter_idle_by_iqr(df, "Origin Stop Idle (mins)", iqr_mult); _check_empty_and_stop(df, "origin dwell IQR")
                df = filter_idle_by_iqr(df, "Destination Stop Idle (mins)", iqr_mult); _check_empty_and_stop(df, "destination dwell IQR")
                if rm_tt_outliers: df = df[df["Travel Time"] <= df["Travel Time"].quantile(0.85)]; _check_empty_and_stop(df, "travel time percentile")
                st.write(f"Final trips after filtering: {len(df)}")

                df["O_Depart_DT"] = pd.to_datetime(df["Origin Stop Departure"], errors='coerce')
                df["Travel Date_dt"] = pd.to_datetime(df["Travel Date"], format="%m/%d/%Y", errors='coerce').dt.date
                df["Time of Day Numeric"] = calculate_time_of_day_adjusted_vectorized(df["O_Depart_DT"], df["Travel Date_dt"])
                df["Time of Day Label"] = df["Time of Day Numeric"].apply(format_time_of_day_label)
                st.session_state.filtered_df = df.reset_index(drop=True)

    if not st.session_state.filtered_df.empty:
        df_final = st.session_state.filtered_df
        st.header(f"📊 Analysis Results: {len(df_final)} Trips")
        disp_copy = df_final.copy()
        dt_cols = ["Origin Stop Arrival", "Origin Stop Departure", "Last Ping In Origin DateTime", "Destination Stop Entry", "Destination Stop Departure"]
        for c in dt_cols: disp_copy[c] = pd.to_datetime(disp_copy[c], errors='coerce').dt.strftime("%m/%d/%y %I:%M %p").fillna("N/A")
        st.dataframe(disp_copy.drop(columns=["O_Depart_DT", "Travel Date_dt", "Time of Day Numeric"], errors='ignore'), use_container_width=True)

        st.subheader("Summary Metrics")
        st.dataframe(pd.DataFrame({
            "Metric": ["Total Trips", "Distinct Vehicles", "Avg Origin Dwell", "Avg Dest Dwell", "Avg Distance", "Avg Travel Time"],
            "Value": [len(df_final), df_final["Vehicle no."].nunique(), f"{_safe_round(df_final['Origin Stop Idle (mins)'].mean())} min", f"{_safe_round(df_final['Destination Stop Idle (mins)'].mean())} min", f"{_safe_round(df_final['Actual Distance (miles)'].mean(), 3)} mi", f"{_safe_round(df_final['Travel Time'].mean())} min"]
        }), use_container_width=True)

        st.subheader("Distribution Plots")
        _plot_dist(df_final, "Travel Time", "Travel Time (minutes)", "Travel Time")
        _plot_dist(df_final, "Origin Stop Idle (mins)", "Dwell Time (minutes)", "Origin Dwell")
        _plot_dist(df_final, "Destination Stop Idle (mins)", "Dwell Time (minutes)", "Destination Dwell") # ADDED

        st.subheader("Time of Day Scatter Plots")
        min_tod, max_tod = df_final["Time of Day Numeric"].min() - 1, df_final["Time of Day Numeric"].max() + 1
        x_axis_tod = alt.X("Time of Day Numeric:Q", title="Time of Day (Origin Departure)", scale=alt.Scale(domain=[max(0,min_tod), max_tod], clamp=True), axis=alt.Axis(labelAngle=-45, tickCount=12))
        _plot_scatter_tod(df_final, "Travel Time", "Travel Time (mins)", x_axis_tod)
        _plot_scatter_tod(df_final, "Origin Stop Idle (mins)", "Origin Dwell (mins)", x_axis_tod)
        _plot_scatter_tod(df_final, "Destination Stop Idle (mins)", "Destination Dwell (mins)", x_axis_tod) # ADDED

        st.subheader("Snap-to-Roads Map (Single Trip)")
        if not google_maps_api_key: st.warning("Enter a Google Maps API Key to enable map feature.")
        else:
            options = [f"Trip (Idx {r.Index}) | Bus: {r._1} | Travel: {r._12:.1f}m" for r in df_final.head(100).itertuples()]
            if options:
                choice_label = st.selectbox("Select trip to display (first 100):", options)
                if choice_label:
                    chosen_idx = int(choice_label.split("Idx ")[1].split(")")[0])
                    chosen_row = df_final.loc[chosen_idx]
                    start, end = pd.to_datetime(chosen_row["Last Ping In Origin DateTime"]), pd.to_datetime(chosen_row["Destination Stop Entry"])
                    if pd.notna(start) and pd.notna(end):
                        pings = st.session_state.path_df[(st.session_state.path_df["AssetNo"] == chosen_row["Vehicle no."]) & (st.session_state.path_df["DateTime"].between(start, end))].sort_values("DateTime")
                        coords = list(zip(pings["Lat"], pings["Lon"]))
                        if coords:
                            with st.spinner("Snapping to roads..."): snapped = snap_to_roads(coords, google_maps_api_key)
                            embed_snapped_polyline_map(coords, snapped, origin_stop_info, dest_stop_info, r_orig, r_dest, google_maps_api_key)
                        else: st.warning("No GPS pings found for the selected trip segment.")
                    else: st.warning("Trip missing timestamps for map.")

if __name__ == "__main__":
    main()