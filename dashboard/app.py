import csv
import json
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import serial
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WiSense AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(30, 80, 130, 0.18),
                transparent 35%
            ),
            #07111f;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.20);
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .system-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .system-subtitle {
        color: #94a3b8;
        margin-bottom: 20px;
    }

    .status-connected {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.4);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 14px;
    }

    .status-disconnected {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.4);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_DIR / "models"

DEFAULT_MODEL = (
    MODELS_DIR
    / "random_forest_csi.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PORT = "/dev/cu.usbmodem1101"

BAUD_RATE = 921600

WINDOW_SECONDS = 2.0

MIN_PACKETS = 20

MAX_BUFFER_PACKETS = 1200


CLASS_NAMES = {
    0: "EMPTY",
    1: "STANDING",
    2: "WALKING",
    3: "SITTING"
}


# ============================================================
# CSI PARSER
# ============================================================

def parse_csi_line(line):

    try:

        row = next(
            csv.reader([line])
        )

        if not row:
            return None

        if row[0] != "CSI_DATA":
            return None

        # RSSI in Espressif CSI format
        rssi = float(row[3])

        # Last CSV field contains:
        # [0,0,-1,2,...]
        values = json.loads(
            row[-1]
        )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        if len(values) < 2:
            return None

        if len(values) % 2 != 0:

            values = values[:-1]

        first = values[0::2]
        second = values[1::2]

        amplitude = np.sqrt(
            first ** 2
            +
            second ** 2
        )

        return {
            "timestamp": time.time(),
            "rssi": rssi,
            "amplitude": amplitude
        }

    except Exception:

        return None


# ============================================================
# SERIAL CSI READER
# ============================================================

class CSIReader:

    def __init__(
        self,
        port,
        baud
    ):

        self.port = port

        self.baud = baud

        self.serial = None

        self.thread = None

        self.running = False

        self.lock = threading.Lock()

        self.timestamps = deque(
            maxlen=MAX_BUFFER_PACKETS
        )

        self.rssi = deque(
            maxlen=MAX_BUFFER_PACKETS
        )

        self.amplitudes = deque(
            maxlen=MAX_BUFFER_PACKETS
        )

        self.total_packets = 0

        self.last_packet_time = 0

        self.last_error = ""


    def start(self):

        try:

            self.serial = serial.Serial(
                self.port,
                self.baud,
                timeout=0.1
            )

            time.sleep(1)

            self.serial.reset_input_buffer()

            self.running = True

            self.thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )

            self.thread.start()

        except Exception as error:

            self.running = False

            self.last_error = str(error)


    def _read_loop(self):

        while self.running:

            try:

                raw = self.serial.readline()

                if not raw:
                    continue

                line = raw.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                packet = parse_csi_line(
                    line
                )

                if packet is None:
                    continue

                with self.lock:

                    self.timestamps.append(
                        packet["timestamp"]
                    )

                    self.rssi.append(
                        packet["rssi"]
                    )

                    self.amplitudes.append(
                        packet["amplitude"]
                    )

                    self.total_packets += 1

                    self.last_packet_time = (
                        packet["timestamp"]
                    )

            except Exception as error:

                self.last_error = str(error)

                time.sleep(0.1)


    def snapshot(self):

        with self.lock:

            return {
                "timestamps": list(
                    self.timestamps
                ),

                "rssi": list(
                    self.rssi
                ),

                "amplitudes": list(
                    self.amplitudes
                ),

                "total_packets":
                    self.total_packets,

                "last_packet_time":
                    self.last_packet_time,

                "error":
                    self.last_error
            }


    def stop(self):

        self.running = False

        if self.serial:

            try:
                self.serial.close()
            except:
                pass


# ============================================================
# CACHED SERIAL CONNECTION
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_reader(
    port,
    baud
):

    reader = CSIReader(
        port,
        baud
    )

    reader.start()

    return reader


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_model(
    path
):

    return joblib.load(
        path
    )


# ============================================================
# FEATURE EXTRACTION
# MUST MATCH TRAINING CODE EXACTLY
# ============================================================

def extract_features(
    window
):

    mean = np.mean(
        window,
        axis=0
    )

    std = np.std(
        window,
        axis=0
    )

    median = np.median(
        window,
        axis=0
    )

    minimum = np.min(
        window,
        axis=0
    )

    maximum = np.max(
        window,
        axis=0
    )

    signal_range = (
        maximum - minimum
    )

    rms = np.sqrt(
        np.mean(
            window ** 2,
            axis=0
        )
    )

    return np.concatenate([
        mean,
        std,
        median,
        signal_range,
        rms
    ])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="system-title">'
    '📡 WiSense AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="system-subtitle">
    Real-Time Device-Free Human Activity Recognition
    using ESP32-S3 Wi-Fi CSI
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ System Control"
)

port = st.sidebar.text_input(
    "ESP32 Serial Port",
    DEFAULT_PORT
)

st.sidebar.text_input(
    "Baud Rate",
    str(BAUD_RATE),
    disabled=True
)

st.sidebar.text_input(
    "Prediction Window",
    f"{WINDOW_SECONDS} seconds",
    disabled=True
)


model_files = list(
    MODELS_DIR.glob("*.joblib")
)

if model_files:

    model_names = [
        x.name
        for x in model_files
    ]

    default_index = 0

    if DEFAULT_MODEL.exists():

        try:

            default_index = model_names.index(
                DEFAULT_MODEL.name
            )

        except:

            pass

    selected_model = st.sidebar.selectbox(
        "Machine Learning Model",
        model_names,
        index=default_index
    )

    model_path = (
        MODELS_DIR
        / selected_model
    )

else:

    st.sidebar.error(
        "No trained model found."
    )

    st.stop()


refresh_ms = st.sidebar.slider(
    "Dashboard refresh rate",
    min_value=300,
    max_value=2000,
    value=700,
    step=100
)


if st.sidebar.button(
    "Reconnect ESP32"
):

    try:

        existing = get_reader(
            port,
            BAUD_RATE
        )

        existing.stop()

    except:

        pass

    get_reader.clear()

    st.rerun()


# ============================================================
# AUTO REFRESH
# ============================================================

st_autorefresh(
    interval=refresh_ms,
    key="wifi_csi_refresh"
)


# ============================================================
# START READER
# ============================================================

reader = get_reader(
    port,
    BAUD_RATE
)

snapshot = reader.snapshot()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model(
        str(model_path)
    )

except Exception as error:

    st.error(
        f"Could not load model: {error}"
    )

    st.stop()


# ============================================================
# MODEL INPUT SIZE
# ============================================================

try:

    expected_features = (
        model.n_features_in_
    )

except:

    expected_features = (
        model.named_steps[
            list(
                model.named_steps.keys()
            )[-1]
        ].n_features_in_
    )


EXPECTED_SUBCARRIERS = (
    expected_features // 5
)


# ============================================================
# CONNECTION STATUS
# ============================================================

now = time.time()

last_packet = snapshot[
    "last_packet_time"
]

connected = (
    last_packet > 0
    and
    now - last_packet < 3
)


if connected:

    st.markdown(
        '<span class="status-connected">'
        '● ESP32 CONNECTED'
        '</span>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<span class="status-disconnected">'
        '● WAITING FOR ESP32 CSI'
        '</span>',
        unsafe_allow_html=True
    )


# ============================================================
# PACKET RATE
# ============================================================

timestamps = np.asarray(
    snapshot["timestamps"],
    dtype=float
)


if len(timestamps):

    packets_5_sec = np.sum(
        timestamps >= now - 5
    )

    packet_rate = (
        packets_5_sec / 5
    )

else:

    packet_rate = 0


# ============================================================
# CURRENT RSSI
# ============================================================

if snapshot["rssi"]:

    current_rssi = (
        snapshot["rssi"][-1]
    )

else:

    current_rssi = 0


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

prediction = None

confidence = 0

probabilities = np.zeros(4)


if len(timestamps):

    recent_indices = np.where(
        timestamps
        >=
        now - WINDOW_SECONDS
    )[0]

    recent_amplitudes = []

    for index in recent_indices:

        amp = snapshot[
            "amplitudes"
        ][index]

        if (
            len(amp)
            >= EXPECTED_SUBCARRIERS
        ):

            recent_amplitudes.append(
                amp[
                    :EXPECTED_SUBCARRIERS
                ]
            )


    if (
        len(recent_amplitudes)
        >= MIN_PACKETS
    ):

        window = np.asarray(
            recent_amplitudes,
            dtype=np.float32
        )

        features = extract_features(
            window
        ).reshape(
            1,
            -1
        )

        prediction = int(
            model.predict(
                features
            )[0]
        )


        if hasattr(
            model,
            "predict_proba"
        ):

            raw_probabilities = (
                model.predict_proba(
                    features
                )[0]
            )

            classes = getattr(
                model,
                "classes_",
                np.arange(
                    len(
                        raw_probabilities
                    )
                )
            )

            probabilities = (
                np.zeros(4)
            )

            for class_value, prob in zip(
                classes,
                raw_probabilities
            ):

                probabilities[
                    int(class_value)
                ] = prob

            confidence = (
                probabilities[
                    prediction
                ]
                * 100
            )


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

if (
    "prediction_history"
    not in st.session_state
):

    st.session_state.prediction_history = []


if (
    "last_prediction_saved"
    not in st.session_state
):

    st.session_state.last_prediction_saved = 0


if (
    prediction is not None
    and
    now
    -
    st.session_state.last_prediction_saved
    >= 1
):

    st.session_state.prediction_history.append({

        "timestamp":
            time.strftime(
                "%H:%M:%S"
            ),

        "unix":
            now,

        "prediction":
            prediction,

        "confidence":
            confidence,

        "empty":
            probabilities[0] * 100,

        "standing":
            probabilities[1] * 100,

        "walking":
            probabilities[2] * 100,

        "sitting":
            probabilities[3] * 100
    })

    st.session_state.prediction_history = (
        st.session_state
        .prediction_history[-100:]
    )

    st.session_state.last_prediction_saved = (
        now
    )


recent_history = (
    st.session_state
    .prediction_history[-5:]
)


if recent_history:

    mean_probs = np.array([
        [
            item["empty"],
            item["standing"],
            item["walking"],
            item["sitting"]
        ]

        for item in recent_history
    ]).mean(
        axis=0
    )

    smoothed_prediction = int(
        np.argmax(
            mean_probs
        )
    )

    smoothed_confidence = (
        mean_probs[
            smoothed_prediction
        ]
    )

else:

    smoothed_prediction = prediction

    smoothed_confidence = confidence


# ============================================================
# TOP METRICS
# ============================================================

if smoothed_prediction is None:

    activity_name = "WAITING"

else:

    activity_name = CLASS_NAMES[
        smoothed_prediction
    ]


col1, col2, col3, col4, col5 = st.columns(
    5
)


with col1:

    st.metric(
        "Detected Activity",
        activity_name
    )


with col2:

    st.metric(
        "Confidence",
        f"{smoothed_confidence:.1f}%"
    )


with col3:

    st.metric(
        "RSSI",
        f"{current_rssi:.0f} dBm"
    )


with col4:

    st.metric(
        "CSI Packet Rate",
        f"{packet_rate:.1f}/s"
    )


with col5:

    st.metric(
        "Total Packets",
        f"{snapshot['total_packets']:,}"
    )


# ============================================================
# ACTIVITY PROBABILITIES
# ============================================================

st.subheader(
    "🧠 Real-Time Activity Probability"
)


probability_fig = go.Figure()


probability_fig.add_trace(
    go.Bar(

        x=[
            "Empty",
            "Standing",
            "Walking",
            "Sitting"
        ],

        y=probabilities * 100,

        text=[
            f"{x:.1f}%"
            for x
            in probabilities * 100
        ],

        textposition="auto"
    )
)


probability_fig.update_layout(

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    yaxis_title="Probability (%)",

    yaxis_range=[0, 100],

    height=320,

    margin=dict(
        l=40,
        r=20,
        t=20,
        b=40
    )
)


st.plotly_chart(
    probability_fig,
    use_container_width=True
)


# ============================================================
# LIVE CSI AMPLITUDE GRAPH
# ============================================================

st.subheader(
    "📈 Live Wi-Fi CSI Amplitude"
)


amps = snapshot[
    "amplitudes"
]


if amps:

    graph_amps = amps[-400:]

    graph_times = timestamps[
        -len(graph_amps):
    ]


    mean_amplitude = np.array([
        np.mean(x)
        for x in graph_amps
    ])


    relative_time = (
        graph_times
        -
        graph_times[-1]
    )


    csi_fig = go.Figure()


    csi_fig.add_trace(
        go.Scatter(

            x=relative_time,

            y=mean_amplitude,

            mode="lines",

            name="CSI amplitude"
        )
    )


    csi_fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title="Time relative to now (seconds)",

        yaxis_title="Mean CSI amplitude",

        height=420,

        margin=dict(
            l=40,
            r=20,
            t=20,
            b=40
        )
    )


    st.plotly_chart(
        csi_fig,
        use_container_width=True
    )

else:

    st.info(
        "Waiting for CSI packets..."
    )


# ============================================================
# CSI SUBCARRIER HEATMAP
# ============================================================

st.subheader(
    "🌐 CSI Subcarrier Heatmap"
)


valid_heatmap = []

heat_times = []


for t, amp in zip(
    snapshot["timestamps"][-150:],
    snapshot["amplitudes"][-150:]
):

    if (
        len(amp)
        >= EXPECTED_SUBCARRIERS
    ):

        valid_heatmap.append(
            amp[
                :EXPECTED_SUBCARRIERS
            ]
        )

        heat_times.append(t)


if valid_heatmap:

    heat = np.asarray(
        valid_heatmap
    )


    heat_times = (
        np.asarray(heat_times)
        -
        heat_times[-1]
    )


    heatmap_fig = go.Figure(
        data=go.Heatmap(

            z=heat.T,

            x=heat_times,

            y=np.arange(
                EXPECTED_SUBCARRIERS
            ),

            colorbar=dict(
                title="Amplitude"
            )
        )
    )


    heatmap_fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title="Time relative to now (seconds)",

        yaxis_title="Subcarrier",

        height=500,

        margin=dict(
            l=40,
            r=20,
            t=20,
            b=40
        )
    )


    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )


# ============================================================
# RSSI GRAPH
# ============================================================

st.subheader(
    "📶 Wi-Fi RSSI"
)


if snapshot["rssi"]:

    rssi_values = np.asarray(
        snapshot["rssi"][-400:]
    )

    rssi_times = timestamps[
        -len(rssi_values):
    ]

    rssi_times = (
        rssi_times
        -
        rssi_times[-1]
    )


    rssi_fig = go.Figure()


    rssi_fig.add_trace(
        go.Scatter(

            x=rssi_times,

            y=rssi_values,

            mode="lines",

            name="RSSI"
        )
    )


    rssi_fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title="Time relative to now (seconds)",

        yaxis_title="RSSI (dBm)",

        height=320
    )


    st.plotly_chart(
        rssi_fig,
        use_container_width=True
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.subheader(
    "🕒 Activity Prediction History"
)


history = (
    st.session_state
    .prediction_history
)


if history:

    history_df = pd.DataFrame(
        history
    )


    history_df[
        "Activity"
    ] = history_df[
        "prediction"
    ].map(
        CLASS_NAMES
    )


    display_df = history_df[
        [
            "timestamp",
            "Activity",
            "confidence",
            "empty",
            "standing",
            "walking",
            "sitting"
        ]
    ].copy()


    display_df.columns = [

        "Time",

        "Activity",

        "Confidence %",

        "Empty %",

        "Standing %",

        "Walking %",

        "Sitting %"
    ]


    display_df = (
        display_df
        .iloc[::-1]
        .head(20)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "Prediction history will appear here."
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🔬 System Information"
)


sys1, sys2, sys3, sys4 = st.columns(
    4
)


with sys1:

    st.write(
        "**Microcontroller**"
    )

    st.write(
        "ESP32-S3 N16R8"
    )


with sys2:

    st.write(
        "**CSI Processing**"
    )

    st.write(
        f"{EXPECTED_SUBCARRIERS} subcarriers"
    )


with sys3:

    st.write(
        "**Model**"
    )

    st.write(
        model_path.name
    )


with sys4:

    st.write(
        "**Window**"
    )

    st.write(
        f"{WINDOW_SECONDS} seconds"
    )


# ============================================================
# ERROR INFORMATION
# ============================================================

if snapshot["error"]:

    with st.expander(
        "ESP32 / Serial Debug Information"
    ):

        st.code(
            snapshot["error"]
        )
        