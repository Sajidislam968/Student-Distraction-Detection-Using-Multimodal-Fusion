"""
=====================================================
FocusMonitor Streamlit Dashboard
=====================================================

Interactive dashboard for:
- Real-time / latest focus state
- Session analytics
- Focus vs distraction time
- Distraction analysis
- Activity, vision, audio, and app usage summaries
- Session history reports

Run from FocusMonitor root:
    streamlit run dashboard/dashboard.py

Expected project structure:
FocusMonitor/
│
├── dashboard/
│   └── dashboard.py
│
├── fusion_dataset/
│   └── fusion_live_dataset.csv
│
├── session_history/
│   └── study_session_history.csv
│
└── fusion_engine/
    └── model/
        ├── training_report.txt
        ├── confusion_matrix.csv
        └── feature_importance.csv
=====================================================
"""

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="FocusMonitor Dashboard",
    page_icon="🎯",
    layout="wide"
)


# =====================================================
# Paths
# =====================================================

CURRENT_FILE = os.path.abspath(__file__)
DASHBOARD_DIR = os.path.dirname(CURRENT_FILE)
BASE_DIR = os.path.dirname(DASHBOARD_DIR)

FUSION_DATASET_PATH = os.path.join(
    BASE_DIR,
    "fusion_dataset",
    "fusion_live_dataset.csv"
)

SESSION_HISTORY_PATH = os.path.join(
    BASE_DIR,
    "session_history",
    "study_session_history.csv"
)

TRAINING_REPORT_PATH = os.path.join(
    BASE_DIR,
    "fusion_engine",
    "model",
    "training_report.txt"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "fusion_engine",
    "model",
    "feature_importance.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "fusion_engine",
    "model",
    "confusion_matrix.csv"
)


# =====================================================
# Constants
# =====================================================

VALID_STATES = [
    "Focused",
    "Distracted",
    "Neutral",
    "Absent"
]

STATE_TO_NUMERIC = {
    "Absent": 0,
    "Distracted": 1,
    "Neutral": 2,
    "Focused": 3,
}

DEFAULT_ROW_INTERVAL_SECONDS = 2.0


# =====================================================
# Utility Functions
# =====================================================

def format_seconds(seconds):
    try:
        seconds = int(float(seconds))
    except Exception:
        seconds = 0

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def parse_hhmmss(value):
    try:
        parts = str(value).split(":")
        if len(parts) != 3:
            return 0.0

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])

        return float(hours * 3600 + minutes * 60 + seconds)

    except Exception:
        return 0.0


def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()

        return pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip"
        )

    except Exception as e:
        st.error(f"Could not read CSV file: {path}")
        st.exception(e)
        return pd.DataFrame()


def safe_to_numeric(series, default=0.0):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(default)


def normalize_state(value):
    value = str(value).strip()

    if value in VALID_STATES:
        return value

    return "Neutral"


def choose_decision_column(df):
    """
    Current fusion_live_dataset.csv has final_label.
    Future versions may have final_decision.
    This function supports both safely.
    """

    if "final_decision" in df.columns:
        return "final_decision"

    if "final_label" in df.columns:
        return "final_label"

    if "auto_label" in df.columns:
        return "auto_label"

    return None


def estimate_row_interval_seconds(df):
    if "timestamp" not in df.columns or len(df) < 3:
        return DEFAULT_ROW_INTERVAL_SECONDS

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    diffs = timestamps.diff().dt.total_seconds()
    diffs = diffs.dropna()
    diffs = diffs[(diffs > 0) & (diffs <= 30)]

    if len(diffs) == 0:
        return DEFAULT_ROW_INTERVAL_SECONDS

    median_interval = float(diffs.median())

    if median_interval <= 0:
        return DEFAULT_ROW_INTERVAL_SECONDS

    return median_interval


def get_latest_session_summary(history_df):
    if history_df.empty:
        return {}

    return history_df.iloc[-1].to_dict()


def calculate_time_distribution_from_live(df, decision_column):
    """
    Estimates time distribution from live fusion log.
    Uses median timestamp gap. If timestamps are not usable,
    assumes 2 seconds per row.
    """

    if df.empty or decision_column is None:
        return {
            "Focused": 0.0,
            "Distracted": 0.0,
            "Neutral": 0.0,
            "Absent": 0.0,
        }

    interval = estimate_row_interval_seconds(df)

    clean_states = df[decision_column].apply(normalize_state)

    time_distribution = {
        "Focused": 0.0,
        "Distracted": 0.0,
        "Neutral": 0.0,
        "Absent": 0.0,
    }

    counts = clean_states.value_counts()

    for state in VALID_STATES:
        time_distribution[state] = float(
            counts.get(state, 0) * interval
        )

    return time_distribution


def calculate_focus_score(time_distribution):
    focused = float(time_distribution.get("Focused", 0.0))
    distracted = float(time_distribution.get("Distracted", 0.0))
    neutral = float(time_distribution.get("Neutral", 0.0))
    absent = float(time_distribution.get("Absent", 0.0))

    total = focused + distracted + neutral + absent

    if total <= 0:
        return 0.0

    score = (
        (focused * 1.0)
        + (neutral * 0.5)
        - (distracted * 0.7)
        - (absent * 1.0)
    ) / total

    score = max(0.0, min(1.0, score))

    return round(score * 100, 2)


def get_latest_state(live_df, history_df):
    latest = {
        "final_decision": "No Data",
        "raw_prediction": "No Data",
        "confidence": 0.0,
        "source": "No Data",
    }

    if not history_df.empty:
        last_session = history_df.iloc[-1]

        latest["final_decision"] = str(
            last_session.get("final_decision", "No Data")
        )

        latest["raw_prediction"] = str(
            last_session.get("last_fusion_prediction", "No Data")
        )

        latest["confidence"] = float(
            last_session.get("last_fusion_confidence", 0.0)
        )

        latest["source"] = "Latest saved session history"

    if not live_df.empty:
        decision_column = choose_decision_column(live_df)
        last_row = live_df.iloc[-1]

        if decision_column is not None:
            latest["live_label"] = normalize_state(
                last_row.get(decision_column, "Neutral")
            )
        else:
            latest["live_label"] = "No Data"

        latest["app_label"] = str(
            last_row.get("app_label", "Unknown")
        )

        latest["auto_label"] = str(
            last_row.get("auto_label", "Unknown")
        )

        latest["auto_reason"] = str(
            last_row.get("auto_reason", "")
        )

        latest["timestamp"] = str(
            last_row.get("timestamp", "")
        )

        latest["source"] = "Live fusion log + latest saved history"

    return latest


def make_metric_card(label, value, help_text=None):
    st.metric(
        label=label,
        value=value,
        help=help_text
    )


def build_text_report(live_df, history_df, time_distribution, focus_score):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    latest_state = get_latest_state(live_df, history_df)

    lines = []
    lines.append("FocusMonitor Dashboard Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Generated At: {now}")
    lines.append("")
    lines.append("Latest Status")
    lines.append("-" * 60)
    lines.append(f"Final Decision       : {latest_state.get('final_decision', 'No Data')}")
    lines.append(f"Raw XGBoost Prediction: {latest_state.get('raw_prediction', 'No Data')}")
    lines.append(f"Raw XGBoost Confidence: {latest_state.get('confidence', 0.0)}%")
    lines.append(f"Latest Live Label     : {latest_state.get('live_label', 'No Data')}")
    lines.append(f"Latest App Label      : {latest_state.get('app_label', 'No Data')}")
    lines.append(f"Latest Auto Label     : {latest_state.get('auto_label', 'No Data')}")
    lines.append(f"Latest Auto Reason    : {latest_state.get('auto_reason', 'No Data')}")
    lines.append("")
    lines.append("Time Distribution")
    lines.append("-" * 60)

    for state in VALID_STATES:
        lines.append(f"{state:<12}: {format_seconds(time_distribution.get(state, 0.0))}")

    lines.append("")
    lines.append(f"Focus Score: {focus_score}%")
    lines.append("")

    if not live_df.empty:
        lines.append("Fusion Dataset")
        lines.append("-" * 60)
        lines.append(f"Rows   : {len(live_df)}")
        lines.append(f"Columns: {len(live_df.columns)}")
        lines.append("")

        if "auto_reason" in live_df.columns:
            lines.append("Top Auto Reasons")
            lines.append("-" * 60)
            top_reasons = live_df["auto_reason"].value_counts().head(10)

            for reason, count in top_reasons.items():
                lines.append(f"{reason}: {count}")

            lines.append("")

    if not history_df.empty:
        lines.append("Session History")
        lines.append("-" * 60)
        lines.append(f"Total Saved Sessions: {len(history_df)}")

    return "\n".join(lines)


# =====================================================
# Data Load
# =====================================================

live_df = safe_read_csv(FUSION_DATASET_PATH)
history_df = safe_read_csv(SESSION_HISTORY_PATH)
feature_importance_df = safe_read_csv(FEATURE_IMPORTANCE_PATH)
confusion_matrix_df = safe_read_csv(CONFUSION_MATRIX_PATH)


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🎯 FocusMonitor")
st.sidebar.caption("AI-Based Multimodal Study Distraction Detection Dashboard")

st.sidebar.markdown("---")

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh dashboard",
    value=False
)

refresh_interval = st.sidebar.slider(
    "Refresh interval seconds",
    min_value=3,
    max_value=30,
    value=5,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Sources")

st.sidebar.write("Fusion Log:")
st.sidebar.code(FUSION_DATASET_PATH)

st.sidebar.write("Session History:")
st.sidebar.code(SESSION_HISTORY_PATH)

st.sidebar.markdown("---")

if st.sidebar.button("Refresh Now"):
    st.rerun()

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()


# =====================================================
# Header
# =====================================================

st.title("🎯 AI-Based Multimodal Study Distraction Detection Dashboard")

st.caption(
    "Real-time focus monitoring, multimodal fusion results, session analytics, "
    "performance reports, and study behavior insights."
)

if live_df.empty and history_df.empty:
    st.warning(
        "No data found yet. Run integrated_monitor.py first, then refresh this dashboard."
    )


# =====================================================
# Main Latest Status
# =====================================================

latest_state = get_latest_state(
    live_df,
    history_df
)

decision_column = choose_decision_column(live_df)
time_distribution = calculate_time_distribution_from_live(
    live_df,
    decision_column
)

focus_score = calculate_focus_score(
    time_distribution
)

total_time = sum(time_distribution.values())

st.markdown("## 1. Live Study Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    make_metric_card(
        "Final Decision",
        latest_state.get("final_decision", latest_state.get("live_label", "No Data"))
    )

with col2:
    make_metric_card(
        "Raw XGBoost",
        latest_state.get("raw_prediction", "No Data")
    )

with col3:
    make_metric_card(
        "Confidence",
        f"{latest_state.get('confidence', 0.0)}%"
    )

with col4:
    make_metric_card(
        "Focus Score",
        f"{focus_score}%"
    )

latest_decision = str(
    latest_state.get(
        "final_decision",
        latest_state.get("live_label", "No Data")
    )
)

if latest_decision == "Focused":
    st.success("Current interpreted state: Focused")
elif latest_decision == "Distracted":
    st.error("Current interpreted state: Distracted")
elif latest_decision == "Absent":
    st.warning("Current interpreted state: Absent")
elif latest_decision == "Neutral":
    st.info("Current interpreted state: Neutral")
else:
    st.info("Current interpreted state: No Data")


# =====================================================
# Latest Multimodal Snapshot
# =====================================================

st.markdown("## 2. Latest Multimodal Snapshot")

if not live_df.empty:
    latest_row = live_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.subheader("Vision")
        st.write(f"Face Detected: {latest_row.get('face_detected', 'N/A')}")
        st.write(f"Head Status: {latest_row.get('head_status', 'N/A')}")
        st.write(f"Head Away: {latest_row.get('head_away', 'N/A')}")
        st.write(f"Eye Status: {latest_row.get('eye_status', 'N/A')}")
        st.write(f"Drowsy: {latest_row.get('drowsy', 'N/A')}")

    with c2:
        st.subheader("Behavior")
        st.write(f"Keyboard Active: {latest_row.get('keyboard_active', 'N/A')}")
        st.write(f"Mouse Active: {latest_row.get('mouse_active', 'N/A')}")
        st.write(f"App Label: {latest_row.get('app_label', 'N/A')}")
        st.write(f"App Confidence: {latest_row.get('app_confidence', 'N/A')}%")

    with c3:
        st.subheader("Audio")
        st.write(f"Speech Status: {latest_row.get('speech_status', 'N/A')}")
        st.write(f"Audio Label: {latest_row.get('audio_label', 'N/A')}")
        st.write(f"Study Probability: {latest_row.get('audio_study_probability', 'N/A')}%")

    with c4:
        st.subheader("Object / Context")
        st.write(f"Phone Detected: {latest_row.get('phone_detected', 'N/A')}")
        st.write(f"Detected Objects: {latest_row.get('detected_objects', '')}")
        st.write(f"Auto Label: {latest_row.get('auto_label', 'N/A')}")
        st.write(f"Reason: {latest_row.get('auto_reason', 'N/A')}")

else:
    st.info("No fusion dataset rows available yet.")


# =====================================================
# Current Session Summary
# =====================================================

st.markdown("## 3. Current Session Analytics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    make_metric_card(
        "Focused Time",
        format_seconds(time_distribution.get("Focused", 0.0))
    )

with col2:
    make_metric_card(
        "Distracted Time",
        format_seconds(time_distribution.get("Distracted", 0.0))
    )

with col3:
    make_metric_card(
        "Neutral Time",
        format_seconds(time_distribution.get("Neutral", 0.0))
    )

with col4:
    make_metric_card(
        "Absent Time",
        format_seconds(time_distribution.get("Absent", 0.0))
    )

with col5:
    make_metric_card(
        "Total Logged Time",
        format_seconds(total_time)
    )

if total_time > 0:
    focused_percentage = round(
        (time_distribution.get("Focused", 0.0) / total_time) * 100,
        2
    )

    distracted_percentage = round(
        (time_distribution.get("Distracted", 0.0) / total_time) * 100,
        2
    )

    st.progress(
        focused_percentage / 100.0,
        text=f"Focused Percentage: {focused_percentage}%"
    )

    st.progress(
        distracted_percentage / 100.0,
        text=f"Distracted Percentage: {distracted_percentage}%"
    )


# =====================================================
# Graphs / Charts
# =====================================================

st.markdown("## 4. Graphs and Performance Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Time Distribution")

    distribution_df = pd.DataFrame({
        "State": list(time_distribution.keys()),
        "Seconds": list(time_distribution.values())
    })

    fig, ax = plt.subplots()
    ax.bar(
        distribution_df["State"],
        distribution_df["Seconds"]
    )
    ax.set_xlabel("State")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Focused / Distracted / Neutral / Absent Time")
    st.pyplot(fig)

with chart_col2:
    st.subheader("Focus State Ratio")

    non_zero_df = distribution_df[distribution_df["Seconds"] > 0]

    if not non_zero_df.empty:
        fig, ax = plt.subplots()
        ax.pie(
            non_zero_df["Seconds"],
            labels=non_zero_df["State"],
            autopct="%1.1f%%"
        )
        ax.set_title("State Percentage")
        st.pyplot(fig)
    else:
        st.info("No time distribution available yet.")


if not live_df.empty and decision_column is not None:
    st.subheader("Focus State Timeline")

    timeline_df = live_df.copy()
    timeline_df["state"] = timeline_df[decision_column].apply(normalize_state)
    timeline_df["state_value"] = timeline_df["state"].map(STATE_TO_NUMERIC)

    if "timestamp" in timeline_df.columns:
        timeline_df["timestamp"] = pd.to_datetime(
            timeline_df["timestamp"],
            errors="coerce"
        )

    fig, ax = plt.subplots()

    if "timestamp" in timeline_df.columns and timeline_df["timestamp"].notna().any():
        ax.plot(
            timeline_df["timestamp"],
            timeline_df["state_value"],
            marker="o",
            linewidth=1
        )
        ax.set_xlabel("Time")
    else:
        ax.plot(
            timeline_df.index,
            timeline_df["state_value"],
            marker="o",
            linewidth=1
        )
        ax.set_xlabel("Row")

    ax.set_ylabel("State")
    ax.set_title("Final State Timeline")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Absent", "Distracted", "Neutral", "Focused"])
    st.pyplot(fig)


# =====================================================
# Distraction Analysis
# =====================================================

st.markdown("## 5. Distraction and Behavior Analysis")

if not live_df.empty:
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

    drowsy_count = int(
        safe_to_numeric(
            live_df.get("drowsy", pd.Series(dtype=float))
        ).sum()
    )

    phone_count = int(
        safe_to_numeric(
            live_df.get("phone_detected", pd.Series(dtype=float))
        ).sum()
    )

    head_away_count = int(
        safe_to_numeric(
            live_df.get("head_away", pd.Series(dtype=float))
        ).sum()
    )

    speech_count = int(
        safe_to_numeric(
            live_df.get("speech_status", pd.Series(dtype=float))
        ).sum()
    )

    no_face_count = 0

    if "face_detected" in live_df.columns:
        no_face_count = int(
            (safe_to_numeric(live_df["face_detected"]) == 0).sum()
        )

    with analysis_col1:
        st.metric("Drowsy Rows", drowsy_count)
        st.metric("Phone Detected Rows", phone_count)

    with analysis_col2:
        st.metric("Head Away Rows", head_away_count)
        st.metric("No Face Rows", no_face_count)

    with analysis_col3:
        st.metric("Speech Detected Rows", speech_count)
        st.metric("Total Fusion Rows", len(live_df))

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.subheader("App Label Distribution")

        if "app_label" in live_df.columns:
            app_counts = live_df["app_label"].fillna("Unknown").value_counts()

            fig, ax = plt.subplots()
            ax.bar(
                app_counts.index.astype(str),
                app_counts.values
            )
            ax.set_xlabel("App Label")
            ax.set_ylabel("Count")
            ax.set_title("Application Classification")
            st.pyplot(fig)
        else:
            st.info("app_label column not found.")

    with detail_col2:
        st.subheader("Top Auto Reasons")

        if "auto_reason" in live_df.columns:
            reason_counts = live_df["auto_reason"].fillna("Unknown").value_counts().head(10)
            st.dataframe(
                reason_counts.rename_axis("Reason").reset_index(name="Count"),
                use_container_width=True
            )
        else:
            st.info("auto_reason column not found.")

else:
    st.info("No live fusion data available for distraction analysis.")


# =====================================================
# Session History
# =====================================================

st.markdown("## 6. Study Session History")

if not history_df.empty:
    display_history = history_df.copy()

    st.dataframe(
        display_history.tail(20),
        use_container_width=True
    )

    st.subheader("Previous Session Focus Trend")

    trend_df = display_history.copy()

    for column in [
        "focused_time",
        "distracted_time",
        "neutral_time",
        "absent_time"
    ]:
        if column in trend_df.columns:
            trend_df[column] = safe_to_numeric(trend_df[column], 0.0)

    if "session_end_time" in trend_df.columns:
        x_values = trend_df["session_end_time"].astype(str)
    else:
        x_values = trend_df.index.astype(str)

    fig, ax = plt.subplots()

    if "focused_time" in trend_df.columns:
        ax.plot(
            x_values,
            trend_df["focused_time"],
            marker="o",
            label="Focused"
        )

    if "distracted_time" in trend_df.columns:
        ax.plot(
            x_values,
            trend_df["distracted_time"],
            marker="o",
            label="Distracted"
        )

    if "neutral_time" in trend_df.columns:
        ax.plot(
            x_values,
            trend_df["neutral_time"],
            marker="o",
            label="Neutral"
        )

    if "absent_time" in trend_df.columns:
        ax.plot(
            x_values,
            trend_df["absent_time"],
            marker="o",
            label="Absent"
        )

    ax.set_xlabel("Session")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Session History Trend")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

else:
    st.info("No session history found yet.")



# =====================================================
# Daily and Weekly Analytics
# =====================================================

st.markdown("## 7. Daily and Weekly Analytics")

if not history_df.empty and "session_end_time" in history_df.columns:
    analytics_df = history_df.copy()

    analytics_df["session_end_time"] = pd.to_datetime(
        analytics_df["session_end_time"],
        errors="coerce"
    )

    analytics_df = analytics_df.dropna(
        subset=["session_end_time"]
    )

    for column in [
        "duration_seconds",
        "focused_time",
        "distracted_time",
        "neutral_time",
        "absent_time"
    ]:
        if column in analytics_df.columns:
            analytics_df[column] = safe_to_numeric(
                analytics_df[column],
                0.0
            )
        else:
            analytics_df[column] = 0.0

    if not analytics_df.empty:
        analytics_df["date"] = analytics_df["session_end_time"].dt.date
        analytics_df["week"] = analytics_df["session_end_time"].dt.to_period("W").astype(str)

        daily_df = analytics_df.groupby("date", as_index=False)[
            [
                "duration_seconds",
                "focused_time",
                "distracted_time",
                "neutral_time",
                "absent_time"
            ]
        ].sum()

        weekly_df = analytics_df.groupby("week", as_index=False)[
            [
                "duration_seconds",
                "focused_time",
                "distracted_time",
                "neutral_time",
                "absent_time"
            ]
        ].sum()

        daily_df["focus_percentage"] = daily_df.apply(
            lambda row: round(
                (row["focused_time"] / row["duration_seconds"]) * 100,
                2
            ) if row["duration_seconds"] > 0 else 0.0,
            axis=1
        )

        weekly_df["focus_percentage"] = weekly_df.apply(
            lambda row: round(
                (row["focused_time"] / row["duration_seconds"]) * 100,
                2
            ) if row["duration_seconds"] > 0 else 0.0,
            axis=1
        )

        daily_tab, weekly_tab = st.tabs([
            "Daily Analytics",
            "Weekly Analytics"
        ])

        with daily_tab:
            st.dataframe(
                daily_df,
                use_container_width=True
            )

            fig, ax = plt.subplots()
            ax.plot(
                daily_df["date"].astype(str),
                daily_df["focus_percentage"],
                marker="o"
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Focus Percentage")
            ax.set_title("Daily Focus Percentage")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        with weekly_tab:
            st.dataframe(
                weekly_df,
                use_container_width=True
            )

            fig, ax = plt.subplots()
            ax.bar(
                weekly_df["week"].astype(str),
                weekly_df["focus_percentage"]
            )
            ax.set_xlabel("Week")
            ax.set_ylabel("Focus Percentage")
            ax.set_title("Weekly Focus Percentage")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    else:
        st.info("Session history exists, but valid session_end_time values were not found.")

else:
    st.info("Daily and weekly analytics will appear after session history is available.")

# =====================================================
# Model Performance
# =====================================================

st.markdown("## 8. Model Performance and Fusion Engine")

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("Feature Importance")

    if not feature_importance_df.empty:
        if "feature" in feature_importance_df.columns and "importance" in feature_importance_df.columns:
            top_features = feature_importance_df.head(15)

            fig, ax = plt.subplots()
            ax.barh(
                top_features["feature"].astype(str),
                safe_to_numeric(top_features["importance"])
            )
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Top Fusion Model Features")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.dataframe(
                feature_importance_df,
                use_container_width=True
            )
        else:
            st.dataframe(
                feature_importance_df,
                use_container_width=True
            )
    else:
        st.info("feature_importance.csv not found yet.")

with perf_col2:
    st.subheader("Confusion Matrix")

    if not confusion_matrix_df.empty:
        st.dataframe(
            confusion_matrix_df,
            use_container_width=True
        )
    else:
        st.info("confusion_matrix.csv not found yet.")

if os.path.exists(TRAINING_REPORT_PATH):
    with st.expander("View XGBoost Training Report"):
        try:
            with open(
                TRAINING_REPORT_PATH,
                "r",
                encoding="utf-8"
            ) as f:
                st.text(f.read())
        except Exception as e:
            st.error(f"Could not read training report: {e}")


# =====================================================
# Report Download
# =====================================================

st.markdown("## 9. Performance Report")

report_text = build_text_report(
    live_df,
    history_df,
    time_distribution,
    focus_score
)

st.download_button(
    label="Download Dashboard Report",
    data=report_text,
    file_name=f"focusmonitor_dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)

with st.expander("Preview Report"):
    st.text(report_text)


# =====================================================
# Raw Data
# =====================================================

st.markdown("## 10. Raw Data Viewer")

tab1, tab2 = st.tabs([
    "Fusion Dataset",
    "Session History"
])

with tab1:
    if not live_df.empty:
        st.dataframe(
            live_df.tail(200),
            use_container_width=True
        )
    else:
        st.info("No fusion dataset available.")

with tab2:
    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True
        )
    else:
        st.info("No session history available.")
