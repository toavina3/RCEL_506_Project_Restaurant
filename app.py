import json
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Restaurant Hourly Revenue by Category",
    page_icon="🍽️",
    layout="wide",
)

required_cols = ["id", "createdTime", "lineItems.elements"]


# Categorize menu items by name
def categorize_item(name):
    name = str(name).strip().lower()

    if any(word in name for word in [
        "cafe", "coffee", "espresso", "latte", "americano",
        "chocolate caliente", "tea", "té", "atole"
    ]):
        return "Hot drinks"

    if any(word in name for word in [
        "agua", "soda", "coca", "coke", "pepsi", "horchata",
        "limonada", "lemonade", "jugo", "juice", "refresco",
        "jarrito", "fanta", "sprite", "jamaica", "drink",
        "paleta", "milkshake", "smoothie"
    ]):
        return "Cold drinks"

    if any(word in name for word in [
        "taco", "tacos", "quesabirria", "quesabirrias",
        "tacos al vapor", "tacos tamaulipas"
    ]):
        return "Tacos"

    if any(word in name for word in [
        "desayuno", "breakfast", "omelette", "omelet",
        "huevo", "huevos", "pancake", "waffle", "chilaquiles",
        "huevos rancheros"
    ]):
        return "Breakfast"

    if any(word in name for word in [
        "hamburguesa", "burger", "quesadilla", "enchilada",
        "burrito", "torta", "sopa", "caldo", "milanesa",
        "fajita", "pollo", "carne", "bistec", "asada",
        "chicken", "beef", "fish", "pescado", "comida",
        "plato", "menudo", "pozole", "empalmes",
        "arroz", "frijoles", "guacamole", "nachos"
    ]):
        return "Main dishes"

    return "Other"


@st.cache_data
def parse_line_items(df_orders):
    """Extract line items, revenue, and menu categories."""
    rows = []
    parse_errors = 0

    for _, row in df_orders.iterrows():
        order_id = row["id"]
        created_time = row["createdTime"]
        line_items_raw = row.get("lineItems.elements", None)

        if not isinstance(line_items_raw, str) or not line_items_raw.strip():
            continue

        try:
            items = json.loads(line_items_raw)

            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                if not item.get("isRevenue", True) or item.get("refunded", False):
                    continue

                item_name = item.get("name", "Unknown Item")
                price = item.get("price", 0)

                if price is None:
                    price = 0

                price = float(price) / 100.0

                category = "Uncategorized"
                item_obj = item.get("item", {})

                if isinstance(item_obj, dict) and "category" in item_obj:
                    cat_data = item_obj["category"]

                    if isinstance(cat_data, dict):
                        category = cat_data.get("name", "Uncategorized")
                    elif isinstance(cat_data, str):
                        category = cat_data

                elif "category" in item:
                    category = item["category"]

                if not category or category == "Uncategorized":
                    category = categorize_item(item_name)

                rows.append({
                    "order_id": order_id,
                    "createdTime": created_time,
                    "item_name": item_name,
                    "category": category,
                    "item_revenue": price,
                })

        except (json.JSONDecodeError, TypeError, ValueError):
            parse_errors += 1
            continue

    if parse_errors:
        st.warning(f"Could not parse line items in {parse_errors} orders.")

    return pd.DataFrame(rows)


@st.cache_data
def fetch_weather_data(start_date, end_date, lat=29.6914242, lon=-95.2193984):
    """Fetch hourly weather data from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": [
            "temperature_2m",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m",
        ],
        "timezone": "America/Chicago",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df_weather = pd.DataFrame(data["hourly"])
    df_weather["datetime"] = pd.to_datetime(df_weather["time"])
    df_weather["date"] = df_weather["datetime"].dt.date
    df_weather["hour"] = df_weather["datetime"].dt.hour

    return df_weather


st.title("🍽️ Hourly Revenue per Menu Category & Weather Analysis")

# Sidebar - Data Loading
st.sidebar.header("1. Load Data")
uploaded_file = st.sidebar.file_uploader("Upload orders.csv", type=["csv"])

if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file, low_memory=False)
else:
    st.info("Please upload your `orders.csv` file in the sidebar to begin.")
    st.stop()

missing_cols = [col for col in required_cols if col not in df1.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# Data Preparation & Timezone Conversion
df1["createdTime"] = (
    pd.to_datetime(df1["createdTime"], unit="ms", utc=True)
    .dt.tz_convert("America/Chicago")
)

# Parse Line Items
df_items = parse_line_items(df1)

if df_items.empty:
    st.error("No valid line item revenue data could be parsed from the file.")
    st.stop()

df_items["createdTime"] = pd.to_datetime(df_items["createdTime"], utc=True)
df_items["createdTime"] = df_items["createdTime"].dt.tz_convert("America/Chicago")
df_items["date"] = df_items["createdTime"].dt.date
df_items["hour"] = df_items["createdTime"].dt.hour
df_items["day_of_week"] = df_items["createdTime"].dt.day_name()

# Aggregate Hourly Category Revenue
df_category_hourly = (
    df_items.groupby(["date", "day_of_week", "hour", "category"])["item_revenue"]
    .sum()
    .reset_index()
)

# Fetch Weather
start_date = df_items["date"].min()
end_date = df_items["date"].max()

try:
    with st.spinner("Fetching weather data for date range..."):
        df_weather = fetch_weather_data(start_date, end_date)
except (requests.RequestException, ValueError, KeyError) as e:
    st.error(f"Unable to fetch weather data: {e}")
    st.stop()

# Merge Sales Data with Weather Data
df_merged = pd.merge(
    df_category_hourly,
    df_weather,
    on=["date", "hour"],
    how="inner",
)

if df_merged.empty:
    st.error("No matching sales and weather data found.")
    st.stop()

# Sidebar - Interactive Filters
st.sidebar.header("2. Filter Parameters")

day_options = [
    "All Days", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
]
selected_day = st.sidebar.selectbox("Select Day of Week", day_options)

selected_hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=(0, 23),
)

temp_min = float(df_merged["temperature_2m"].min())
temp_max = float(df_merged["temperature_2m"].max())

selected_temp_range = st.sidebar.slider(
    "Temperature (°C)",
    min_value=round(temp_min, 1),
    max_value=round(temp_max, 1),
    value=(round(temp_min, 1), round(temp_max, 1)),
)

precip_min = float(df_merged["precipitation"].min())
precip_max = float(df_merged["precipitation"].max())

selected_precip_range = st.sidebar.slider(
    "Precipitation (mm)",
    min_value=round(precip_min, 1),
    max_value=round(precip_max, 1),
    value=(round(precip_min, 1), round(precip_max, 1)),
)

wind_min = float(df_merged["wind_speed_10m"].min())
wind_max = float(df_merged["wind_speed_10m"].max())

selected_wind_range = st.sidebar.slider(
    "Wind Speed (km/h)",
    min_value=round(wind_min, 1),
    max_value=round(wind_max, 1),
    value=(round(wind_min, 1), round(wind_max, 1)),
)

# Apply Filters
filtered_df = df_merged[
    (df_merged["hour"] >= selected_hour_range[0])
    & (df_merged["hour"] <= selected_hour_range[1])
    & (df_merged["temperature_2m"] >= selected_temp_range[0])
    & (df_merged["temperature_2m"] <= selected_temp_range[1])
    & (df_merged["precipitation"] >= selected_precip_range[0])
    & (df_merged["precipitation"] <= selected_precip_range[1])
    & (df_merged["wind_speed_10m"] >= selected_wind_range[0])
    & (df_merged["wind_speed_10m"] <= selected_wind_range[1])
]

if selected_day != "All Days":
    filtered_df = filtered_df[
        filtered_df["day_of_week"] == selected_day
    ]

st.subheader(
    f"Revenue per Category from {selected_hour_range[0]}:00 "
    f"to {selected_hour_range[1]}:00 ({selected_day})"
)
)

# Metrics Summary
col1, col2, col3, col4 = st.columns(4)

total_filtered_rev = filtered_df["item_revenue"].sum()
unique_dates_count = filtered_df["date"].nunique()
avg_hourly_rev = (
    total_filtered_rev / unique_dates_count if unique_dates_count > 0 else 0
)

col1.metric("Total Revenue", f"${total_filtered_rev:,.2f}")
col2.metric("Matching Days Count", f"{unique_dates_count}")
col3.metric("Avg Hourly Revenue", f"${avg_hourly_rev:,.2f}")

if not filtered_df.empty:
    top_cat = (
        filtered_df.groupby("category")["item_revenue"].sum().idxmax()
    )
    col4.metric("Top Category", top_cat)
else:
    col4.metric("Top Category", "N/A")

st.markdown("---")

# Category Revenue Aggregation
cat_summary = (
    filtered_df.groupby("category")["item_revenue"]
    .agg(["sum", "mean"])
    .reset_index()
)

cat_summary.columns = [
    "Category",
    "Total Revenue ($)",
    "Avg Revenue per Hour ($)",
]

cat_summary = cat_summary.sort_values(
    by="Total Revenue ($)", ascending=False
)

if not cat_summary.empty:
    fig = px.bar(
        cat_summary,
        x="Category",
        y="Total Revenue ($)",
        text_auto=".2f",
        title=(
            f"Total Revenue per Category "
            f"from {selected_hour_range[0]}:00 to {selected_hour_range[1]}:00"
        ),
        color="Category",
    )

    fig.update_layout(
        xaxis_title="Menu Category",
        yaxis_title="Revenue ($)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Breakdown")
    st.dataframe(cat_summary, use_container_width=True)
else:
    st.warning(
        "No transactions found matching the selected weather and time filters."
    )
