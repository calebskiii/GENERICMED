import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------
# APP SETTINGS
# ------------------------------------------------------------
st.set_page_config(page_title="Offline Handwriting Recognition", page_icon="✍️", layout="wide")

CHARS = [
    ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', ':', ';', '?',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]
IDX_TO_CHAR = {i + 1: char for i, char in enumerate(CHARS)}
BLANK_IDX = 0
NUM_CLASSES = len(CHARS) + 1

# ------------------------------------------------------------
# MODEL: matches branchCNN_TCN_CTC in the supplied notebook
# ------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        weights = F.adaptive_avg_pool2d(x, 1)
        weights = F.relu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.net(x))


class TemporalConv1d(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )

    def forward(self, x):
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = TemporalConv1d(channels, dilation)
        self.conv2 = TemporalConv1d(channels, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        return F.relu(x + residual)


class branchCNN_TCN_CTC(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.shared_stem = nn.Sequential(
            ConvBlock(1, 32), nn.MaxPool2d(2, 2),
            ConvBlock(32, 64), nn.MaxPool2d(2, 2),
            ConvBlock(64, 128), nn.MaxPool2d(2, 2),
            ConvBlock(128, 256)
        )
        self.cnn_skip_proj = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        self.tcn_input = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.tcn = nn.Sequential(
            TCNBlock(128, dilation=1, dropout=dropout),
            TCNBlock(128, dilation=2, dropout=dropout),
            TCNBlock(128, dilation=4, dropout=dropout),
            TCNBlock(128, dilation=8, dropout=dropout)
        )
        self.tcn_norm = nn.BatchNorm1d(128)
        self.skip_norm = nn.BatchNorm1d(128)
        self.temporal_fusion_norm = nn.BatchNorm1d(128)
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1)
        )
        self.fusion_norm = nn.BatchNorm1d(128)
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    @staticmethod
    def output_lengths(widths):
        return torch.div(widths, 8, rounding_mode="floor")

    def forward(self, x):
        shared = self.shared_stem(x)
        cnn_mean = shared.mean(dim=2)
        cnn_max = shared.amax(dim=2)
        skip_input = torch.cat([cnn_mean, cnn_max], dim=1)
        cnn_skip = self.cnn_skip_proj(skip_input)
        temporal_input = cnn_mean + cnn_max
        temporal_features = self.tcn_input(temporal_input)
        temporal_features = self.tcn(temporal_features)
        temporal_features = self.tcn_norm(temporal_features)
        cnn_skip = self.skip_norm(cnn_skip)
        temporal_features = self.temporal_fusion_norm(temporal_features)
        fusion_input = torch.cat([cnn_skip, temporal_features], dim=1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        fused_features = (1.0 - gate) * cnn_skip + gate * temporal_features
        fused_features = self.fusion_norm(fused_features)
        fused_features = F.relu(fused_features)
        fused_features = self.fusion_dropout(fused_features)
        logits = self.classifier(fused_features)
        return logits.permute(2, 0, 1).contiguous()


@st.cache_resource(show_spinner="Loading trained recognition model...")
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = branchCNN_TCN_CTC(num_classes=NUM_CLASSES, dropout=0.1).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def prepare_image(image):
    """Matches the notebook's custom-image path: grayscale, Otsu, height 128, [-1, 1]."""
    image = image.convert("L")
    image_array = np.asarray(image)
    _, binarized = cv2.threshold(
        image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    processed = Image.fromarray(binarized)
    width, height = processed.size
    new_height = 128
    new_width = max(32, int(round(width * new_height / height)))
    new_width = min(new_width, 2048)
    resized = processed.resize((new_width, new_height), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
    return processed, resized, tensor, new_width


def decode_with_confidence(logits, output_length):
    probabilities = torch.softmax(logits.float(), dim=2)
    predictions = probabilities.argmax(dim=2)
    decoded, confidence_values = [], []
    current_index, current_values = None, []

    for t in range(min(output_length, predictions.shape[0])):
        index = int(predictions[t, 0].item())
        if index == BLANK_IDX:
            if current_index is not None:
                decoded.append(IDX_TO_CHAR[current_index])
                confidence_values.append(float(np.mean(current_values)))
                current_index, current_values = None, []
        elif index == current_index:
            current_values.append(float(probabilities[t, 0, index].item()))
        else:
            if current_index is not None:
                decoded.append(IDX_TO_CHAR[current_index])
                confidence_values.append(float(np.mean(current_values)))
            current_index = index
            current_values = [float(probabilities[t, 0, index].item())]

    if current_index is not None:
        decoded.append(IDX_TO_CHAR[current_index])
        confidence_values.append(float(np.mean(current_values)))
    return "".join(decoded), confidence_values


@torch.inference_mode()
def recognize(image, model, device):
    processed, resized, tensor, new_width = prepare_image(image)
    tensor = tensor.to(device)
    started = time.perf_counter()
    logits = model(tensor)
    widths = torch.tensor([new_width], device=device, dtype=torch.long)
    output_length = int(model.output_lengths(widths).clamp(min=1, max=logits.size(0))[0].item())
    prediction, confidences = decode_with_confidence(logits, output_length)
    elapsed = (time.perf_counter() - started) * 1000
    return processed, resized, prediction, confidences, elapsed


# ------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------
st.title("✍️ Offline Handwriting Recognition")
st.caption("Write a word or sentence below, then submit it to the trained CNN–TCN–CTC model.")

with st.sidebar:
    st.header("Model setup")
    checkpoint = st.text_input(
        "Checkpoint file",
        value="cnn_tcn_gated_fixedlr_v2.pth",
        help="Place the .pth checkpoint in this app folder, or enter its full file path."
    )
    st.markdown("---")
    st.write("**Supported input**")
    st.write("English handwriting using the model's trained character set: uppercase and lowercase letters, digits, spaces, and supported punctuation.")
    st.caption("The app runs locally. The model is loaded once and is not retrained when you press Recognize.")

checkpoint_path = Path(checkpoint).expanduser()
if not checkpoint_path.is_file():
    st.warning("Model checkpoint not found. Put `cnn_tcn_gated_fixedlr_v2.pth` beside `app.py`, or enter the correct checkpoint path in the sidebar.")
    st.info("The interface is ready, but recognition will work only after the trained .pth checkpoint is available.")
    model, device = None, None
else:
    try:
        model, device = load_model(str(checkpoint_path.resolve()))
        st.success(f"Trained model loaded successfully · {device}")
    except Exception as error:
        model, device = None, None
        st.error(f"Could not load checkpoint: {error}")

st.subheader("Write your handwriting")
st.write("Use your mouse, trackpad, touchscreen, or stylus. Write on one line where possible.")

if "canvas_reset" not in st.session_state:
    st.session_state.canvas_reset = 0

left, right = st.columns([3, 1])
with left:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=4,
        stroke_color="#111111",
        background_color="#FFFFFF",
        height=220,
        width=900,
        drawing_mode="freedraw",
        key=f"handwriting_canvas_{st.session_state.canvas_reset}",
        display_toolbar=True,
        update_streamlit=True,
        return_image_data=True,
    )
with right:
    st.markdown("**Tips**")
    st.write("• Write naturally and clearly.")
    st.write("• Keep the sentence on one line.")
    st.write("• Leave small spaces between words.")
    st.write("• Avoid writing outside the canvas.")

button_col1, button_col2 = st.columns([1, 2])
with button_col1:
    if st.button("🧹 Clear handwriting", use_container_width=True):
        st.session_state.canvas_reset += 1
        st.session_state.pop("last_result", None)
        st.rerun()
with button_col2:
    recognize_clicked = st.button("🔍 Recognize handwriting", type="primary", use_container_width=True)

if recognize_clicked:
    if canvas_result.image_data is None:
        st.error("Please write something on the canvas first.")
    elif model is None:
        st.error("The trained checkpoint is not loaded. Check the checkpoint path in the sidebar.")
    else:
        rgba = canvas_result.image_data.astype(np.uint8)
        rgb_pixels = rgba[:, :, :3]
        has_ink = np.any(np.min(rgb_pixels, axis=2) < 245)
        if not has_ink:
            st.error("The canvas is empty. Please write a word or sentence first.")
        else:
            # Canvas background is white; preserve the drawn strokes.
            user_image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
            with st.spinner("Recognizing handwriting..."):
                try:
                    processed, resized, prediction, confidences, elapsed = recognize(user_image, model, device)
                    st.session_state.last_result = {
                        "prediction": prediction,
                        "confidences": confidences,
                        "elapsed": elapsed,
                        "processed": processed,
                        "resized": resized,
                    }
                except Exception as error:
                    st.error(f"Recognition failed: {error}")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.divider()
    st.subheader("Recognition result")
    st.markdown("**Predicted text**")
    if result["prediction"].strip():
        st.markdown(
            f'<div style="padding:18px;border:1px solid #888;border-radius:10px;font-size:26px;min-height:65px;overflow-wrap:anywhere">{result["prediction"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("The model did not produce a readable character sequence. Try writing more clearly or using a larger canvas area.")
    st.caption(f"Inference time: {result['elapsed']:.1f} ms")
    if result["confidences"]:
        st.markdown("**Character-level confidence**")
        confidence_columns = st.columns(min(8, len(result["confidences"])))
        for index, (char, confidence) in enumerate(zip(result["prediction"], result["confidences"])):
            with confidence_columns[index % len(confidence_columns)]:
                label = "space" if char == " " else char
                st.metric(label, f"{confidence * 100:.1f}%")
    with st.expander("View processed model input"):
        st.image(result["processed"], caption="Grayscale image after Otsu thresholding", use_container_width=True)
        st.image(result["resized"], caption="Resized model input (height 128)", use_container_width=True)

st.divider()
st.caption("Research demonstration only. Predictions on panelists' handwriting are live demonstrations and should not be combined with the formal IAM test-set CER/WER results unless separately evaluated and documented.")
