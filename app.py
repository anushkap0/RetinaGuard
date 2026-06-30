import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
IMG_SIZE    = 224

CLASS_INFO = {
    'No_DR':          ('🟢', 'No Diabetic Retinopathy', 'No signs of DR detected. Regular annual screening recommended.'),
    'Mild':           ('🟡', 'Mild DR',                 'Microaneurysms present. Monitor every 6–12 months.'),
    'Moderate':       ('🟠', 'Moderate DR',             'More severe than mild. Ophthalmologist visit recommended.'),
    'Severe':         ('🔴', 'Severe DR',               'Extensive abnormalities. Urgent specialist referral needed.'),
    'Proliferate_DR': ('🚨', 'Proliferative DR',        'Most advanced stage. Immediate treatment required.'),
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('retina_best_v4.keras')

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img_pil):
    img = img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 255.0) * 2.0 - 1.0          # [-1, 1] — matches training
    return np.expand_dims(arr, axis=0), np.array(img)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def make_gradcam(img_array, model):
    mobilenet  = model.get_layer('mobilenetv2_1.00_224')
    grad_model = tf.keras.Model(
        inputs  = mobilenet.input,
        outputs = [mobilenet.get_layer('out_relu').output, mobilenet.output]
    )

    with tf.GradientTape() as tape:
        conv_out, mob_out = grad_model(img_array, training=False)
        tape.watch(conv_out)

        x = model.get_layer('global_average_pooling2d')(mob_out)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout')(x)
        x = model.get_layer('dense_1')(x)
        x = model.get_layer('dropout_1')(x)
        preds = model.get_layer('dense_2')(x)

        pred_idx      = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_idx]

    grads        = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_idx, preds.numpy()[0]

def build_gradcam_overlay(heatmap, img_display, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    superimposed   = np.clip(
        heatmap_colored * alpha + (img_display / 255.0) * (1 - alpha), 0, 1
    )
    return heatmap_resized, superimposed

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Fundus_photograph_of_normal_right_eye.jpg/800px-Fundus_photograph_of_normal_right_eye.jpg",
             caption="Sample retinal fundus image")
    st.markdown("## 👁️ RetinaScan AI")
    st.markdown("Detect **Diabetic Retinopathy** severity from retinal fundus images using deep learning.")
    st.markdown("---")
    st.markdown("### 📊 DR Severity Scale")
    for cls, (icon, label, _) in CLASS_INFO.items():
        st.markdown(f"{icon} **{label}**")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
- **Model:** MobileNetV2 (fine-tuned)
- **Classes:** 5 DR severity levels  
- **Dataset:** APTOS 2019 (~3,662 images)
- **Val Accuracy:** ~72.7%
- **Val AUC:** 0.946
    """)
    st.markdown("---")
    alpha = st.slider("Grad-CAM overlay intensity", 0.1, 0.9, 0.4, 0.05)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("👁️ RetinaScan AI — Diabetic Retinopathy Detection")
st.markdown("Upload a **retinal fundus image** to detect the severity of Diabetic Retinopathy.")
st.markdown("---")

uploaded = st.file_uploader(
    "📤 Upload Retinal Fundus Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear retinal fundus photograph"
)

if uploaded is not None:
    img_pil = Image.open(uploaded)

    with st.spinner("🔍 Analysing image..."):
        try:
            model = load_model()
            img_array, img_display = preprocess(img_pil)
            heatmap, pred_idx, probs = make_gradcam(img_array, model)
            heatmap_resized, superimposed = build_gradcam_overlay(heatmap, img_display, alpha)

            pred_class = CLASS_NAMES[pred_idx]
            icon, label, advice = CLASS_INFO[pred_class]
            confidence = probs[pred_idx] * 100

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.stop()

    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown("## 🧠 Diagnosis Result")
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        if pred_class == 'No_DR':
            st.success(f"{icon} **{label}**")
        elif pred_class in ['Mild', 'Moderate']:
            st.warning(f"{icon} **{label}**")
        else:
            st.error(f"{icon} **{label}**")
        st.metric("Confidence", f"{confidence:.1f}%")

    with col_res2:
        st.info(f"💡 **Clinical Advice:** {advice}")
        st.markdown(f"**Predicted Class:** `{pred_class}`")

    st.markdown("---")

    # ── Image columns ─────────────────────────────────────────────────────────
    st.markdown("## 🔬 Visual Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(img_display, use_container_width=True)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        heatmap_rgb = (plt.cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        st.image(heatmap_rgb, use_container_width=True)
        st.caption("🔴 Red = regions model focused on most")

    with col3:
        st.markdown("**Overlay**")
        overlay_uint8 = (superimposed * 255).astype(np.uint8)
        st.image(overlay_uint8, use_container_width=True)
        st.caption("Heatmap superimposed on retinal image")

    st.markdown("---")

    # ── Probability chart ─────────────────────────────────────────────────────
    st.markdown("## 📊 Class Probabilities")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['crimson' if i == pred_idx else 'steelblue' for i in range(len(CLASS_NAMES))]
    bars   = ax.bar(CLASS_NAMES, probs * 100, color=colors, edgecolor='black', linewidth=0.5)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{prob*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Confidence (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title('Model Confidence per DR Class', fontsize=13, fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4, label='50% threshold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.caption("⚠️ This tool is for research/educational purposes only and should not replace professional medical diagnosis.")

else:
    # Landing state
    st.markdown("### 👆 Upload an image to get started")
    c1, c2, c3 = st.columns(3)
    c1.info("📤 **Step 1** — Upload a retinal fundus image (JPG/PNG)")
    c2.info("🧠 **Step 2** — AI analyses the image instantly")
    c3.info("📊 **Step 3** — View diagnosis + Grad-CAM heatmap")
    st.markdown("---")
    st.markdown("#### What is Diabetic Retinopathy?")
    st.markdown("""
Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes. 
It's caused by damage to the blood vessels in the retina and is a leading cause of blindness worldwide.
Early detection through regular screening is critical for preventing vision loss.

This AI model analyses retinal fundus photographs and classifies DR severity into 5 levels,
helping prioritise patients who need urgent care.
    """)