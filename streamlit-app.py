import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import requests
import streamlit.components.v1 as components
from math import radians, sin, cos, sqrt, atan2

################################################################################
# 1) Utility: Haversine & Circle checks
################################################################################

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 20902231  # Earth's approximate radius in feet
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

def point_in_circle(point_lat, point_lon, center_lat, center_lon, radius_feet):
    return haversine_distance(point_lat, point_lon, center_lat, center_lon) <= radius_feet

def interpolate_time(t1, t2):
    if t1 is None or t2 is None:
        return None
    diff = (t2 - t1)/2
    midpoint = t1 + diff
    return midpoint.replace(microsecond=0)

################################################################################
# 2) Ping-Based Stop Crossing Analysis
################################################################################

def analyze_stop_crossings(
    stops_df,
    path_df,
    origin_stop,
    destination_stop,
    radius_feet_origin=200,
    radius_feet_destination=200
):
    origin_row = stops_df[stops_df["Stop Name"] == origin_stop].iloc[0]
    origin_lat, origin_lon = origin_row["Lat"], origin_row["Lon"]

    dest_row = stops_df[stops_df["Stop Name"] == destination_stop].iloc[0]
    dest_lat, dest_lon = dest_row["Lat"], dest_row["Lon"]

    results = []
    for bus_id in path_df["Asset No."].unique():
        bus_data = path_df[path_df["Asset No."] == bus_id].copy()
        bus_data = bus_data.sort_values("DateTime")

        inside_origin = False
        origin_arrival = None
        origin_depart  = None

        inside_dest = False
        dest_arrival = None

        for i in range(len(bus_data)-1):
            curr = bus_data.iloc[i]
            nxt  = bus_data.iloc[i+1]

            c_in_ori = point_in_circle(curr["Lat"], curr["Lon"], origin_lat, origin_lon, radius_feet_origin)
            n_in_ori = point_in_circle(nxt["Lat"],  nxt["Lon"],  origin_lat, origin_lon, radius_feet_origin)

            c_in_dst = point_in_circle(curr["Lat"], curr["Lon"], dest_lat, dest_lon, radius_feet_destination)
            n_in_dst = point_in_circle(nxt["Lat"],  nxt["Lon"],  dest_lat, dest_lon, radius_feet_destination)

            # ORIGIN ARRIVAL
            if not inside_origin and c_in_ori:
                inside_origin = True
                origin_arrival = curr["DateTime"]

            # ORIGIN DEPARTURE
            if inside_origin and not n_in_ori:
                inside_origin = False
                origin_depart = interpolate_time(origin_arrival, nxt["DateTime"])

            # DESTINATION ARRIVAL
            if (origin_depart is not None) and (not inside_dest) and c_in_dst:
                inside_dest = True
                dest_arrival = curr["DateTime"]

                # measure distance from origin_depart -> dest_arrival
                trip_slice = bus_data[
                    (bus_data["DateTime"] >= origin_depart) &
                    (bus_data["DateTime"] <= dest_arrival)
                ]
                if len(trip_slice) > 0:
                    dist_start = trip_slice.iloc[0]["Distance Traveled(Miles)"]
                    dist_end   = trip_slice.iloc[-1]["Distance Traveled(Miles)"]
                    actual_distance = dist_end - dist_start
                else:
                    actual_distance = 0.0

                if origin_arrival and origin_depart:
                    origin_idle = (origin_depart - origin_arrival).total_seconds()/60.0
                else:
                    origin_idle = 0.0

                if origin_depart and dest_arrival:
                    travel_time = (dest_arrival - origin_depart).total_seconds()/60.0
                else:
                    travel_time = 0.0

                results.append({
                    "Vehicle no.": bus_id,
                    "Travel Date": (origin_depart or origin_arrival).strftime("%m/%d/%Y"),
                    "Origin Stop": origin_stop,
                    "Destination Stop": destination_stop,
                    "Origin Stop Departure": origin_depart,
                    "Origin Stop Idle (mins)": round(origin_idle,2),
                    "Destination Stop Entry": dest_arrival,
                    "Destination Stop Departure": None,
                    "Destination Stop Idle (mins)": None,
                    "Actual Distance (miles)": round(actual_distance,3),
                    "Travel Time": round(travel_time,2)
                })

            # DESTINATION DEPARTURE
            if inside_dest and not n_in_dst:
                inside_dest = False
                circle_exit = nxt["DateTime"]
                dest_depart = interpolate_time(dest_arrival, circle_exit)

                if results:
                    last = results[-1]
                    if (last["Vehicle no."] == bus_id) and (last["Destination Stop Departure"] is None):
                        last["Destination Stop Departure"] = dest_depart
                        if dest_arrival and dest_depart:
                            d_idle = (dest_depart - dest_arrival).total_seconds()/60.0
                        else:
                            d_idle = 0.0
                        last["Destination Stop Idle (mins)"] = round(d_idle,2)

                origin_arrival = None
                origin_depart  = None
                dest_arrival   = None

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # convert datetimes to string
    for c in ["Origin Stop Departure","Destination Stop Entry","Destination Stop Departure"]:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%m/%d/%Y %I:%M:%S %p")

    df = df[df["Destination Stop Departure"] != ""].copy()

    final_cols = [
        "Vehicle no.","Travel Date","Origin Stop","Destination Stop",
        "Origin Stop Departure","Origin Stop Idle (mins)",
        "Destination Stop Entry","Destination Stop Departure",
        "Destination Stop Idle (mins)",
        "Actual Distance (miles)","Travel Time"
    ]
    df = df[final_cols]
    return df

################################################################################
# 3) Distance Filter
################################################################################

def filter_by_gap_clustering_largest(df, gap_threshold=0.3):
    """Largest cluster approach for 'Actual Distance (miles)'. Remove outliers if there's a big gap."""
    if df.empty:
        return df
    dist_col = "Actual Distance (miles)"
    sorted_df = df.sort_values(by=dist_col)
    distances = sorted_df[dist_col].values

    clusters = []
    curr_cluster = [0]
    for i in range(len(distances)-1):
        curr_val = distances[i]
        next_val = distances[i+1]
        if (next_val - curr_val) >= gap_threshold:
            clusters.append(curr_cluster)
            curr_cluster = [i+1]
        else:
            curr_cluster.append(i+1)
    if curr_cluster:
        clusters.append(curr_cluster)

    best_cluster_idx = None
    best_size = 0
    best_mean = None
    idx_array = sorted_df.index.to_list()
    for c_idx_list in clusters:
        c_vals = [distances[i] for i in c_idx_list]
        c_size = len(c_vals)
        c_mean = np.mean(c_vals)
        if c_size > best_size:
            best_cluster_idx = c_idx_list
            best_size = c_size
            best_mean = c_mean
        elif c_size == best_size:
            if best_mean is None or c_mean < best_mean:
                best_cluster_idx = c_idx_list
                best_mean = c_mean

    if not best_cluster_idx:
        return df

    keep_indexes = [ idx_array[i] for i in best_cluster_idx ]
    newdf = df.loc[keep_indexes].copy()
    return newdf

################################################################################
# 4) Idle Largest Cluster
################################################################################

def filter_idle_by_gap_clustering_largest(df, idle_col, gap_threshold=0.3):
    """We cluster 'idle_col' ascending and keep the largest cluster => remove outlier dwell times."""
    if df.empty:
        return df
    sub_df = df.dropna(subset=[idle_col])
    if sub_df.empty:
        return df

    sorted_sub = sub_df.sort_values(by=idle_col)
    idle_vals  = sorted_sub[idle_col].values
    idx_arr    = sorted_sub.index.to_list()

    if len(idle_vals) < 2:
        return df

    clusters = []
    curr_cluster = [0]
    for i in range(len(idle_vals)-1):
        curr_val = idle_vals[i]
        next_val = idle_vals[i+1]
        if (next_val - curr_val) >= gap_threshold:
            clusters.append(curr_cluster)
            curr_cluster = [i+1]
        else:
            curr_cluster.append(i+1)
    if curr_cluster:
        clusters.append(curr_cluster)

    best_cluster_idx = None
    best_size = 0
    best_mean = None
    for c_idx_list in clusters:
        c_vals = [ idle_vals[i] for i in c_idx_list ]
        c_size = len(c_vals)
        c_mean = np.mean(c_vals)
        if c_size > best_size:
            best_cluster_idx = c_idx_list
            best_size = c_size
            best_mean = c_mean
        elif c_size == best_size:
            if best_mean is None or c_mean < best_mean:
                best_cluster_idx = c_idx_list
                best_mean = c_mean

    if not best_cluster_idx:
        return df

    keep_idx = [ idx_arr[i] for i in best_cluster_idx ]
    filtered = df.loc[keep_idx].copy()
    return filtered

################################################################################
# 5) Idle IQR Approach
################################################################################

def filter_idle_by_iqr(df, idle_col, multiplier=1.5):
    """Removes borderline outliers above Q3 + multiplier*IQR."""
    if df.empty or (idle_col not in df.columns):
        return df
    sub = df.dropna(subset=[idle_col])
    if sub.empty:
        return df

    q1 = sub[idle_col].quantile(0.25)
    q3 = sub[idle_col].quantile(0.75)
    iqr= q3 - q1
    upper_cut = q3 + multiplier*iqr
    newdf = df[df[idle_col].fillna(0) <= upper_cut].copy()
    return newdf

################################################################################
# 6) Snap-to-Roads
################################################################################

def snap_to_roads(coords, api_key):
    if not coords:
        return []
    snapped_points = []
    BATCH_SIZE = 100
    for i in range(0, len(coords), BATCH_SIZE):
        chunk = coords[i:i+BATCH_SIZE]
        path_param = "|".join(f"{c[0]},{c[1]}" for c in chunk)
        url = (
            "https://roads.googleapis.com/v1/snapToRoads"
            f"?path={path_param}&interpolate=false&key={api_key}"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            st.write(f"Snap to Roads error: {resp.text}")
            continue
        data = resp.json()
        points = data.get("snappedPoints", [])
        for p in points:
            loc = p["location"]
            snapped_points.append((loc["latitude"], loc["longitude"]))
    return snapped_points

def embed_snapped_polyline_map(
    original_coords,
    snapped_coords,
    origin_stop_lat, origin_stop_lon,
    dest_stop_lat, dest_stop_lon,
    radius_feet_origin,
    radius_feet_destination,
    api_key
):
    if not snapped_coords:
        st.write("No snapped coords to display.")
        return

    r_m_origin = float(radius_feet_origin)*0.3048
    r_m_dest   = float(radius_feet_destination)*0.3048

    center_lat, center_lon = snapped_coords[0]
    snapped_js = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in snapped_coords)
    orig_js    = ",\n".join(f"{{ lat: {pt[0]}, lng: {pt[1]} }}" for pt in original_coords)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
      <style>
        #map {{
          height: 100%;
          width: 100%;
        }}
        html, body {{
          margin: 0;
          padding: 0;
          height: 100%;
        }}
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

          var originCircle = new google.maps.Circle({{
            strokeColor: "#0000FF",
            strokeOpacity: 0.8,
            strokeWeight: 2,
            fillColor: "#0000FF",
            fillOpacity: 0.1,
            map: map,
            center: {{ lat: {origin_stop_lat}, lng: {origin_stop_lon} }},
            radius: {r_m_origin}
          }});
          var destCircle = new google.maps.Circle({{
            strokeColor: "#00FF00",
            strokeOpacity: 0.8,
            strokeWeight: 2,
            fillColor: "#00FF00",
            fillOpacity: 0.1,
            map: map,
            center: {{ lat: {dest_stop_lat}, lng: {dest_stop_lon} }},
            radius: {r_m_dest}
          }});

          var originMarker = new google.maps.Marker({{
            position: {{ lat: {origin_stop_lat}, lng: {origin_stop_lon} }},
            map: map,
            title: "Origin Stop"
          }});
          var destMarker = new google.maps.Marker({{
            position: {{ lat: {dest_stop_lat}, lng: {dest_stop_lon} }},
            map: map,
            title: "Destination Stop"
          }});

          var snappedCoords = [
            {snapped_js}
          ];
          if (snappedCoords.length > 1) {{
            var snappedRoute = new google.maps.Polyline({{
              path: snappedCoords,
              geodesic: true,
              strokeColor: "#4285F4",
              strokeOpacity: 1.0,
              strokeWeight: 4
            }});
            snappedRoute.setMap(map);
          }}

          var originalCoords = [
            {orig_js}
          ];
          originalCoords.forEach(function(coord) {{
            new google.maps.Marker({{
              position: coord,
              map: map,
              icon: {{
                path: google.maps.SymbolPath.CIRCLE,
                scale: 3,
                fillColor: '#FF0000',
                fillOpacity: 1,
                strokeColor: '#FF0000',
                strokeWeight: 1
              }}
            }});
          }});
        }}
      </script>
      <script async
        src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap">
      </script>
    </body>
    </html>
    """

    components.html(html_code, height=600)

################################################################################
# 7) Main App
################################################################################

def main():
    st.title("Zonar Stop Time Analysis")

    # 1) Provide an input box for the API key
    st.write("**Enter Google Maps API key:**")
    google_maps_api_key = st.text_input("Google Maps API Key", value="", type="password")

    # We'll store final data in session state
    if "filtered_df" not in st.session_state:
        st.session_state["filtered_df"] = None
        st.session_state["stops_df"]    = None
        st.session_state["path_df"]     = None
        st.session_state["origin_stop"] = None
        st.session_state["destination_stop"] = None
        st.session_state["r_feet_origin"] = 200
        st.session_state["r_feet_dest"]   = 200

    stops_file = st.file_uploader("Upload stops classification CSV", type=["csv"])
    path_file  = st.file_uploader("Upload Zonar data CSV", type=["csv"])

    # col1, col2 = st.columns(2)
    # with col1:
    #     start_date = st.date_input("Start date")
    # with col2:
    #     end_date   = st.date_input("End date")

    cA, cB = st.columns(2)
    with cA:
        radius_feet_origin = st.number_input("Origin circle radius (feet)", min_value=1, value=200)
    with cB:
        radius_feet_destination = st.number_input("Destination circle radius (feet)", min_value=1, value=200)

    st.write("**Distance Filtering (experimental)**")
    max_distance_miles = st.number_input("Max distance (leave at 0 for auto calculation):", min_value=0.0, value=0.0)
    gap_threshold_dist  = st.number_input("Gap threshold for distance cluster (only applicable if max=0, determines the gap size between distance clusters)", min_value=0.0, value=0.3)

    # dwell cluster gap
    dwell_gap = st.number_input("Dwell cluster gap threshold (mins)", min_value=0.0, value=0.3)

    # iqr multiplier
    iqr_multiplier = st.number_input("IQR multiplier for dwell times (1.5 typical, increase or decrease this to include or exclude more outliers)", min_value=0.1, value=1.5)

    # remove top 15% travel time?
    remove_tt_15pct = st.checkbox("Remove top 15% Travel Time outliers?", value=False)

    if stops_file and path_file:
        stops_df = pd.read_csv(stops_file, low_memory=False)
        path_df  = pd.read_csv(path_file, low_memory=False)

        if not set(["Stop Name","Lat","Lon"]).issubset(stops_df.columns):
            st.error("Stops CSV must have 'Stop Name','Lat','Lon'")
            return
        if "Stop ID" not in stops_df.columns:
            st.error("Stops CSV missing 'Stop ID'")
            return

        stops_df = stops_df.sort_values("Stop ID").copy()
        stops_df["Stop_Name_ID"] = stops_df["Stop Name"].astype(str) + " (" + stops_df["Stop ID"].astype(str) + ")"

        req_cols = ["Asset No.","Date","Time(EST)","Distance Traveled(Miles)","Lat","Lon"]
        missing = [c for c in req_cols if c not in path_df.columns]
        if missing:
            st.error(f"Path CSV missing columns: {missing}")
            return

        path_df["DateTime"] = pd.to_datetime(path_df["Date"] + " " + path_df["Time(EST)"], errors="coerce")
        # if start_date and end_date and start_date <= end_date:
        #     sdt = datetime.combine(start_date, datetime.min.time())
        #     edt = datetime.combine(end_date, datetime.max.time())
        #     path_df = path_df[ (path_df["DateTime"]>=sdt) & (path_df["DateTime"]<=edt) ]

        path_df = path_df.sort_values(["Asset No.","DateTime"])

        origin_choice = st.selectbox("Origin Stop:", stops_df["Stop_Name_ID"].tolist())
        dest_choice   = st.selectbox("Destination Stop:", stops_df["Stop_Name_ID"].tolist())

        origin_stop = stops_df.loc[stops_df["Stop_Name_ID"]==origin_choice,"Stop Name"].iloc[0]
        dest_stop   = stops_df.loc[stops_df["Stop_Name_ID"]==dest_choice,"Stop Name"].iloc[0]

        if st.button("Analyze Crossings"):
            results_df = analyze_stop_crossings(
                stops_df=stops_df,
                path_df=path_df,
                origin_stop=origin_stop,
                destination_stop=dest_stop,
                radius_feet_origin=radius_feet_origin,
                radius_feet_destination=radius_feet_destination
            )
            if results_df.empty:
                st.warning("No valid trips found.")
                st.session_state["filtered_df"] = None
                return

            # distance filter
            if max_distance_miles > 0:
                final_df = results_df[ results_df["Actual Distance (miles)"] <= max_distance_miles ]
            else:
                final_df = filter_by_gap_clustering_largest(results_df, gap_threshold=gap_threshold_dist)
            if final_df.empty:
                st.warning("All removed by distance filter.")
                st.session_state["filtered_df"] = None
                return

            # largest cluster approach for origin dwell
            final_df = filter_idle_by_gap_clustering_largest(final_df, "Origin Stop Idle (mins)", dwell_gap)
            # then for destination dwell
            final_df = filter_idle_by_gap_clustering_largest(final_df, "Destination Stop Idle (mins)", dwell_gap)
            if final_df.empty:
                st.warning("All removed by largest-cluster dwell approach.")
                st.session_state["filtered_df"] = None
                return

            # IQR approach for origin
            final_df = filter_idle_by_iqr(final_df, "Origin Stop Idle (mins)", multiplier=iqr_multiplier)
            # IQR for destination
            final_df = filter_idle_by_iqr(final_df, "Destination Stop Idle (mins)", multiplier=iqr_multiplier)
            if final_df.empty:
                st.warning("All removed by IQR dwell approach.")
                st.session_state["filtered_df"] = None
                return

            # optional remove top 15% travel time
            if remove_tt_15pct:
                tt_85 = final_df["Travel Time"].quantile(0.85)
                final_df = final_df[ final_df["Travel Time"] <= tt_85 ]
                if final_df.empty:
                    st.warning("All removed by top 15% travel time approach.")
                    st.session_state["filtered_df"] = None
                    return

            if final_df.empty:
                st.warning("No data remains after all filters.")
                st.session_state["filtered_df"] = None
                return

            # parse "Origin Stop Departure" => time-of-day
            def parse_o_depart(x):
                if not x:
                    return None
                fmt = "%m/%d/%Y %I:%M:%S %p"
                try:
                    return datetime.strptime(x, fmt)
                except:
                    return None
            
            final_df["O_Depart_DT"] = final_df["Origin Stop Departure"].apply(parse_o_depart)
            def get_time_of_day(dt_):
                if dt_ is None:
                    return None
                return dt_.hour + dt_.minute/60.0
            final_df["Time of Day"] = final_df["O_Depart_DT"].apply(get_time_of_day)

            st.session_state["filtered_df"]       = final_df
            st.session_state["stops_df"]          = stops_df
            st.session_state["path_df"]           = path_df
            st.session_state["origin_stop"]       = origin_stop
            st.session_state["destination_stop"]  = dest_stop
            st.session_state["r_feet_origin"]     = radius_feet_origin
            st.session_state["r_feet_dest"]       = radius_feet_destination

    # Show final
    final_df = st.session_state.get("filtered_df", None)
    if final_df is not None and not final_df.empty:
        st.write(f"### Final Trips: {len(final_df)}")
        st.dataframe(final_df, use_container_width=True)

        # aggregates
        n_trips = len(final_df)
        n_veh   = final_df["Vehicle no."].nunique()
        avg_o_idle = final_df["Origin Stop Idle (mins)"].mean()
        avg_d_idle = final_df["Destination Stop Idle (mins)"].mean()
        avg_dist   = final_df["Actual Distance (miles)"].mean()
        avg_time   = final_df["Travel Time"].mean()

        st.write("**Summary after filtering**")
        ag = {
            "Total Trips": [n_trips],
            "Distinct Vehicles": [n_veh],
            "Avg Origin Dwell": [round(avg_o_idle,2) if pd.notnull(avg_o_idle) else 0],
            "Avg Dest Dwell": [round(avg_d_idle,2) if pd.notnull(avg_d_idle) else 0],
            "Avg Distance": [round(avg_dist,3) if pd.notnull(avg_dist) else 0],
            "Avg Travel Time": [round(avg_time,2) if pd.notnull(avg_time) else 0]
        }
        st.dataframe(pd.DataFrame(ag), use_container_width=True)

        chart_df = final_df.dropna(subset=["Time of Day"]).copy()

        st.write("### Travel Time vs Time of Day")
        if not chart_df.empty:
            c1 = (
                alt.Chart(chart_df)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("Time of Day:Q", title="Time of Day (0=midnight...23.99=11:59PM)"),
                    y=alt.Y("Travel Time:Q", title="Travel Time (mins)"),
                    tooltip=["Vehicle no.","Travel Time","Time of Day"]
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(c1, use_container_width=True)
        else:
            st.write("No data for travel time chart.")

        st.write("### Origin Dwell vs Time of Day")
        if not chart_df.empty:
            c2 = (
                alt.Chart(chart_df)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("Time of Day:Q", title="Time of Day"),
                    y=alt.Y("Origin Stop Idle (mins):Q", title="Dwell Time (mins)"),
                    tooltip=["Vehicle no.","Origin Stop Idle (mins)","Time of Day"]
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(c2, use_container_width=True)
        else:
            st.write("No data for origin dwell chart.")

        st.write("### Destination Dwell vs Time of Day")
        if not chart_df.empty:
            c3 = (
                alt.Chart(chart_df)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("Time of Day:Q", title="Time of Day"),
                    y=alt.Y("Destination Stop Idle (mins):Q", title="Dwell Time (mins)"),
                    tooltip=["Vehicle no.","Destination Stop Idle (mins)","Time of Day"]
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(c3, use_container_width=True)
        else:
            st.write("No data for dest dwell chart.")

        st.write("### Snap-to-Roads Map (Optional)")
        if google_maps_api_key == "":
            st.warning("Please enter a valid Roads API key above.")
        else:
            # pick a trip
            trip_opts = []
            for idx, row in final_df.iterrows():
                label = (
                    f"Row {idx} | Bus={row['Vehicle no.']} | Dist={row['Actual Distance (miles)']} | "
                    f"OIdle={row['Origin Stop Idle (mins)']} | DIdle={row['Destination Stop Idle (mins)']}"
                )
                trip_opts.append((label, idx))

            if trip_opts:
                choice_label = st.selectbox("Pick a trip:", [t[0] for t in trip_opts])
                chosen_idx   = next(t[1] for t in trip_opts if t[0] == choice_label)
                chosen_row   = final_df.loc[chosen_idx]

                o_depart_str = chosen_row["Origin Stop Departure"]
                d_depart_str = chosen_row["Destination Stop Departure"]
                if not o_depart_str or not d_depart_str:
                    st.warning("Cannot parse times => no map.")
                    return

                fmt = "%m/%d/%Y %I:%M:%S %p"
                try:
                    o_dt = datetime.strptime(o_depart_str, fmt)
                except:
                    o_dt = None
                try:
                    d_dt = datetime.strptime(d_depart_str, fmt)
                except:
                    d_dt = None

                if (o_dt is None) or (d_dt is None):
                    st.warning("Could not parse date times => no map.")
                    return

                stops_df = st.session_state["stops_df"]
                path_df  = st.session_state["path_df"]
                orig_stop= st.session_state["origin_stop"]
                dest_stop= st.session_state["destination_stop"]
                rfo      = st.session_state["r_feet_origin"]
                rfd      = st.session_state["r_feet_dest"]

                bus_id = chosen_row["Vehicle no."]
                bus_data = path_df[path_df["Asset No."]==bus_id].copy()
                mask = (bus_data["DateTime"] >= o_dt) & (bus_data["DateTime"] <= d_dt)
                bus_data = bus_data[mask].sort_values("DateTime")

                coords = []
                for _, rdata in bus_data.iterrows():
                    lat, lon = rdata["Lat"], rdata["Lon"]
                    if pd.notnull(lat) and pd.notnull(lon):
                        coords.append((lat, lon))

                st.write(f"Trip has {len(coords)} raw points.")
                snapped_points = snap_to_roads(coords, google_maps_api_key)

                ori_info = stops_df[stops_df["Stop Name"]==orig_stop].iloc[0]
                des_info = stops_df[stops_df["Stop Name"]==dest_stop].iloc[0]

                embed_snapped_polyline_map(
                    original_coords=coords,
                    snapped_coords=snapped_points,
                    origin_stop_lat=ori_info["Lat"],
                    origin_stop_lon=ori_info["Lon"],
                    dest_stop_lat=des_info["Lat"],
                    dest_stop_lon=des_info["Lon"],
                    radius_feet_origin=rfo,
                    radius_feet_destination=rfd,
                    api_key=google_maps_api_key
                )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
