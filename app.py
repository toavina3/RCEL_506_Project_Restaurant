import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Operations Bottleneck Dashboard", layout="wide"
)
st.title("Restaurant Order Fulfillment & Speed Analytics")


# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # Format hours and ensure correct day ordering
    df["formatted_hour"] = pd.to_datetime(df["hour"], format="%H").dt.strftime(
        "%I:00 %p"
    )
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df["day_of_week"] = pd.Categorical(
        df["day_of_week"], categories=day_order, ordered=True
    )

    return df


df = load_data()

# Sidebar Interactive Filters
st.sidebar.header("Filter Parameters")
selected_days = st.sidebar.multiselect(
    "Select Days of Week",
    options=df["day_of_week"].cat.categories.tolist(),
    default=df["day_of_week"].cat.categories.tolist(),
)

emp_range = st.sidebar.slider(
    "Employee Count On Duty",
    int(df["employee_count"].min()),
    int(df["employee_count"].max()),
    (int(df["employee_count"].min()), int(df["employee_count"].max())),
)

order_types = st.sidebar.multiselect(
    "Order Type",
    options=df["orderType.label"].unique().tolist(),
    default=df["orderType.label"].unique().tolist(),
)

# Apply filters
filtered_df = df[
    (df["day_of_week"].isin(selected_days))
    & (df["employee_count"].between(emp_range[0], emp_range[1]))
    & (df["orderType.label"].isin(order_types))
]

# Section 1: Single Consolidated Visualization (Parallel Coordinates Plot)
st.subheader(
    "1. All-in-One Multi-Parameter Pathway (Parallel Coordinates)"
)
fig_pcp = px.parallel_coordinates(
    filtered_df,
    dimensions=[
        "hour",
        "employee_count",
        "num_items",
        "num_customizations",
        "processing_time_min",
    ],
    color="processing_time_min",
    color_continuous_scale=px.colors.diverging.Tealrose,
    labels={
        "hour": "Hour",
        "employee_count": "Staff",
        "num_items": "Items",
        "num_customizations": "Modifiers",
        "processing_time_min": "Prep Time (min)",
    },
)
st.plotly_chart(fig_pcp, use_container_width=True)

# Section 2: Interactive Operational Heatmap
st.subheader("2. Processing Time Heatmap (Hour vs Day)")
heatmap_data = (
    filtered_df.groupby(["day_of_week", "hour", "formatted_hour"])[
        ["processing_time_min", "employee_count", "id_x"]
    ]
    .agg(
        avg_prep_time=("processing_time_min", "median"),
        avg_employees=("employee_count", "mean"),
        total_orders=("id_x", "nunique"),
    )
    .reset_index()
    .sort_values("hour")
)

fig_heatmap = px.density_heatmap(
    heatmap_data,
    x="formatted_hour",
    y="day_of_week",
    z="avg_prep_time",
    histfunc="avg",
    color_continuous_scale="Reds",
    labels={
        "formatted_hour": "Hour of Day",
        "day_of_week": "Day",
        "avg_prep_time": "Median Prep Time (min)",
    },
)
st.plotly_chart(fig_heatmap, use_container_width=True)
