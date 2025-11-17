# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "19dS6rAzHlGekODz1l2F020D9XMlhNDYS")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
    labels[0]:{"texts":["중국식 냉면 입니다"],
              "videos":["https://www.youtube.com/watch?v=9ZuSRJtpWSE"],
              "images":["data:image/jpeg;base64,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"]},
    labels[1]:{"texts":["짜장면 입니다"],
              "videos":["https://www.youtube.com/watch?v=tQUTkWfHdO8"],
              "images":["data:image/jpeg;base64,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"]},
    labels[2]:{"texts":["짬뽕 입니다"],
              "videos":["https://www.youtube.com/watch?v=3FBKcTumM5w"],
              "images":["https://img-cf.kurly.com/hdims/resize/%3E720x/quality/90/src/shop/data/goodsview/20230803/gv20000714335_1.jpg"]},
    labels[3]:{"texts":["탕수육 입니다"],
              "videos":["https://www.youtube.com/watch?v=e9jHy8AJ4yc"],
              "images":["https://homecuisine.co.kr/files/attach/images/142/737/002/969e9f7dc60d42510c5c0353a58ba701.JPG"]},

}
# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
