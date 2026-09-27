from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, UnidentifiedImageError

from utils.assessment import assess_machine, maintenance_recommendations
from utils.model_utils import FEATURE_COLUMNS, ModelLoadError, find_model_file, load_model
from utils.report_generator import generate_assessment_pdf

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
PAGES = (
    "Dashboard",
    "Machine Analysis",
    "Prediction Details",
    "Maintenance Recommendation",
    "Service Locator",
    "Assessment Report",
    "About",
)

st.set_page_config(
    page_title="Industrial Machine Health Assessment & Failure Prediction",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading trained model...")
def cached_model(model_path: str, modified_ns: int):
    del modified_ns
    return load_model(Path(model_path))


def initialize_state() -> None:
    defaults = {
        "assessment": None,
        "assessment_values": None,
        "assessment_model_filename": None,
        "assessment_image_filename": None,
        "report_pdf": None,
        "service_maps_url": None,
        "service_maps_query": None,
        "image_bytes": None,
        "image_filename": None,
        "image_error": None,
        "image_upload_generation": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_current_model():
    path = find_model_file(MODELS_DIR)
    if path is None:
        return None, None, "No .pkl model found in the models/ folder. Add the trained artifact to enable predictions."
    try:
        model, features = cached_model(str(path.resolve()), path.stat().st_mtime_ns)
        return model, features, None
    except (ModelLoadError, OSError, ValueError) as exc:
        return None, path.name, str(exc)


def validate_and_store_image(uploaded_file) -> None:
    if uploaded_file is None:
        return
    if uploaded_file.size > MAX_IMAGE_BYTES:
        st.session_state.image_bytes = None
        st.session_state.image_filename = None
        st.session_state.image_error = "Image exceeds the 20 MB upload limit."
        return
    try:
        image_bytes = uploaded_file.getvalue()
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Choose a JPG, JPEG, or PNG image.")
            width, height = image.size
        st.session_state.image_bytes = image_bytes
        st.session_state.image_filename = uploaded_file.name
        st.session_state.image_dimensions = (width, height)
        st.session_state.image_error = None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.session_state.image_bytes = None
        st.session_state.image_filename = None
        st.session_state.image_error = f"Image could not be validated: {exc}"


def current_values() -> dict:
    return {
        "Machine Type": st.session_state.get("machine_type", "H"),
        "Air temperature [K]": st.session_state.get("air_temperature"),
        "Process temperature [K]": st.session_state.get("process_temperature"),
        "Rotational speed [rpm]": st.session_state.get("rotational_speed"),
        "Torque [Nm]": st.session_state.get("torque"),
        "Tool wear [min]": st.session_state.get("tool_wear"),
    }


def render_sidebar(model, model_filename, model_error) -> str:
    with st.sidebar:
        st.markdown(
            "<div class='brand-lockup'><div class='brand-mark'>MH</div>"
            "<div><strong>INDUSTRIAL</strong><br>HEALTH SYSTEM</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("CONDITION MONITORING / LOCAL CONSOLE")
        st.markdown("<div class='sidebar-label'>WORKSPACE</div>", unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, key="page_navigation", label_visibility="collapsed")
        st.markdown("<div class='sidebar-label'>MODEL STATUS</div>", unsafe_allow_html=True)
        if model is not None:
            st.success("Model loaded", icon=":material/check_circle:")
            st.caption(f"Active artifact: {model_filename}")
        else:
            st.error("Model not loaded", icon=":material/warning:")
            st.caption(f"Artifact: {model_filename or 'none'}")
            with st.expander("Model status details"):
                st.write(model_error)
        st.caption("Sensor readings are operator-entered. Images are visual reference only.")
    return page


def navigate_to_analysis() -> None:
    st.session_state.page_navigation = "Machine Analysis"


def navigate_to_service_locator() -> None:
    st.session_state.page_navigation = "Service Locator"


def render_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f"<div class='page-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def render_dashboard(model, model_filename, model_error) -> None:
    st.markdown("<div class='page-kicker'>OPERATIONS / CONDITION MONITORING</div>", unsafe_allow_html=True)
    with st.container(key="dashboard-hero", border=True):
        intro, status = st.columns([1.35, 0.65], vertical_alignment="center")
        with intro:
            st.title("Industrial Machine Health Assessment")
            st.write("Machine Failure Prediction & Maintenance Assistant")
            st.caption("Decision support from measured operating parameters and a supplied, trained Decision Tree.")
            st.button(
                "Start machine assessment",
                type="primary",
                icon=":material/arrow_forward:",
                on_click=navigate_to_analysis,
                key="start_assessment",
            )
        with status:
            st.markdown("**SYSTEM READINESS**")
            if model is not None:
                st.success("Prediction model ready", icon=":material/check_circle:")
                st.caption(f"Artifact: {model_filename}")
            else:
                st.warning("Prediction model unavailable", icon=":material/warning:")
                st.caption(model_error)

    st.markdown("### Assessment sequence")
    cards = [
        ("01", "Enter readings", "Provide the five measured model parameters and machine type."),
        ("02", "Assess condition", "Run the supplied classifier and review its output and decision path."),
        ("03", "Plan follow-up", "Review maintenance guidance and download the assessment report."),
        ("04", "Find local service", "Search Google Maps for repair providers near your location."),
    ]
    for column, (number, heading, body) in zip(st.columns(4), cards):
        with column, st.container(border=True):
            st.markdown(f"<span class='step-index'>{number}</span>", unsafe_allow_html=True)
            st.markdown(f"**{heading}**")
            st.caption(body)

    st.markdown("### What the model uses")
    details, image_note = st.columns([1.2, 0.8], gap="large")
    with details:
        with st.container(border=True):
            st.markdown("**Measured parameters**")
            st.write("Air temperature [K] · Process temperature [K] · Rotational speed [rpm] · Torque [Nm] · Tool wear [min]")
            st.caption("Machine type: H, M, or L. These are the complete supported model inputs.")
    with image_note:
        st.warning(
            "Machine images are for operator visual reference only. They do not provide sensor measurements.",
            icon=":material/visibility:",
        )


def render_machine_analysis(model, feature_order, model_filename, model_error) -> None:
    render_header("INPUT / ASSESS", "Machine Analysis", "Enter the measured operating state to run the trained model.")
    if model is None:
        st.error(f"Prediction is disabled. {model_error}")

    parameters, visual_reference = st.columns([1.2, 0.8], gap="large")
    with parameters:
        st.markdown("### 01 / Operating parameters")
        st.caption("Use current measured values. All five numeric fields are required.")
        with st.form("machine_assessment_form", border=True):
            st.selectbox("Machine type", options=["H", "M", "L"], key="machine_type")
            first, second = st.columns(2)
            with first:
                st.number_input("Air temperature [K]", min_value=0.0, value=None, placeholder="Measured value", key="air_temperature", help="Enter the measured value in kelvin.")
                st.number_input("Rotational speed [rpm]", min_value=0.0, value=None, placeholder="Measured value", key="rotational_speed")
                st.number_input("Tool wear [min]", min_value=0.0, value=None, placeholder="Measured value", key="tool_wear")
            with second:
                st.number_input("Process temperature [K]", min_value=0.0, value=None, placeholder="Measured value", key="process_temperature", help="Enter the measured value in kelvin.")
                st.number_input("Torque [Nm]", min_value=0.0, value=None, placeholder="Measured value", key="torque")
            submitted = st.form_submit_button(
                "Run assessment",
                type="primary",
                icon=":material/assessment:",
                width="stretch",
                disabled=model is None,
            )

        if submitted:
            values = current_values()
            if any(values[field] is None for field in FEATURE_COLUMNS[:5]):
                st.error("Enter all five measured numeric parameters before running the assessment.")
            else:
                try:
                    result = assess_machine(model, feature_order, values)
                    st.session_state.assessment = result
                    st.session_state.assessment_values = values
                    st.session_state.assessment_model_filename = model_filename
                    st.session_state.assessment_image_filename = st.session_state.image_filename
                    st.session_state.report_pdf = None
                    st.success("Assessment complete. Open Prediction Details or Maintenance Recommendation for the result.")
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    st.error(f"Assessment could not be completed: {exc}")

    with visual_reference:
        st.markdown("### 02 / Visual reference")
        st.caption("Optional image · JPG, JPEG, or PNG · maximum 20 MB")
        upload_key = f"machine_image_{st.session_state.image_upload_generation}"
        uploaded_file = st.file_uploader(
            "Upload machine image",
            type=["jpg", "jpeg", "png"],
            key=upload_key,
            max_upload_size=20,
        )
        if uploaded_file is not None:
            validate_and_store_image(uploaded_file)
        if st.session_state.image_error:
            st.error(st.session_state.image_error)
        if st.session_state.image_bytes:
            st.image(BytesIO(st.session_state.image_bytes), caption=st.session_state.image_filename, width="stretch")
            width, height = st.session_state.get("image_dimensions", (None, None))
            if width is not None:
                st.caption(f"{st.session_state.image_filename} · {width} × {height} px")
            if st.button("Remove image", icon=":material/delete:", key="remove_machine_image"):
                st.session_state.image_upload_generation += 1
                st.session_state.image_bytes = None
                st.session_state.image_filename = None
                st.session_state.image_dimensions = None
                st.session_state.image_error = None
                st.rerun()
        else:
            st.caption("No image uploaded. Image upload is optional; prediction uses the operating parameters only.")
        st.info("The image cannot determine temperature, speed, torque, or tool wear. Enter those measurements in the operating-parameter form.", icon=":material/info:")


def render_prediction_details(model) -> None:
    render_header("OUTPUT / MODEL EXPLANATION", "Prediction Details", "The assessment output and model-specific explanation for the submitted inputs.")
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("No assessment is available yet. Submit measured parameters on Machine Analysis.")
        return

    first, second, third = st.columns(3)
    first.metric("Failure probability", f"{assessment['failure_probability']:.1%}")
    second.metric("Risk band", assessment["risk_level"])
    third.metric("Model classification", "Failure" if assessment["predicted_failure"] else "No failure")
    if assessment["predicted_failure"]:
        st.error("The model classified these inputs as failure (class 1). Treat this as a decision-support alert.")
    else:
        st.success("The model did not classify these inputs as failure (class 0). This is not a guarantee of safe operation.")
    st.caption(f"Explanation type: {assessment['explanation_source']}")

    if assessment["factors"]:
        if all("rule" in factor for factor in assessment["factors"]):
            for index, factor in enumerate(assessment["factors"], start=1):
                st.markdown(f"**{index}. {factor['rule']}**  ")
                st.caption(f"Submitted value: {factor['value']:g}. This split was traversed by this assessment in the trained decision tree.")
        else:
            labels = [factor["feature"] for factor in assessment["factors"]]
            values = [factor["importance"] for factor in assessment["factors"]]
            figure, axis = plt.subplots(figsize=(8, max(2.5, len(labels) * 0.45)))
            axis.barh(labels[::-1], values[::-1], color="#e66d47")
            axis.set_xlabel("Overall model feature importance")
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.grid(axis="x", alpha=0.2)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            st.caption("Global importance describes the model overall; it does not establish causation for this individual prediction.")
    else:
        st.info("This artifact does not expose a supported feature-level explanation.")

    with st.expander("Inputs used by the model"):
        st.json(assessment["feature_values"])
    st.caption(f"Model artifact: {st.session_state.assessment_model_filename}. Probability comes from the model's predict_proba output; risk bands are application triage labels (20% and 50% cutoffs).")


def render_recommendations() -> None:
    render_header("OUTPUT / NEXT STEPS", "Maintenance Recommendation", "Practical follow-up based on the model result and its available feature explanation.")
    st.markdown("### Find a local repair provider")
    st.caption("Search current business listings by repair type and location using Google Maps.")
    st.button(
        "Open service locator",
        icon=":material/map:",
        on_click=navigate_to_service_locator,
        key="open_service_locator",
    )
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("Run an assessment on Machine Analysis to see maintenance recommendations.")
        return
    if assessment["risk_level"] == "High":
        st.error(f"High-priority review · {assessment['failure_probability']:.1%} model failure probability")
    elif assessment["risk_level"] == "Moderate":
        st.warning(f"Planned review · {assessment['failure_probability']:.1%} model failure probability")
    else:
        st.success(f"Routine monitoring · {assessment['failure_probability']:.1%} model failure probability")
    for number, recommendation in enumerate(maintenance_recommendations(assessment), start=1):
        st.markdown(f"**{number}.** {recommendation}")
    st.caption("These are triage suggestions derived from the model result, not equipment-specific service instructions. Follow manufacturer guidance and site safety procedures.")


def build_maps_search_url(service_type: str, location: str) -> tuple[str, str]:
    query = f"{service_type.strip()} near {location.strip()}"
    url = "https://www.google.com/maps/search/?" + urlencode({"api": "1", "query": query})
    return query, url


def render_service_locator() -> None:
    render_header(
        "LOCAL SUPPORT / GOOGLE MAPS",
        "Service locator",
        "Find nearby repair providers using a location-specific Google Maps search.",
    )
    st.caption("Enter your town, city, or postal code. Google Maps will show the current businesses and contact details for that area.")

    with st.form("service_locator_form", border=True):
        location_column, service_column = st.columns([1, 1], gap="large")
        with location_column:
            st.text_input(
                "Your location",
                placeholder="City, town, or postal code",
                max_chars=120,
                key="service_location",
            )
        with service_column:
            st.text_input(
                "Service or repair type",
                value="industrial machine repair",
                max_chars=100,
                key="service_type",
                help="Choose a general search or specify the equipment or repair service you need.",
            )
        submitted = st.form_submit_button(
            "Find nearby repair shops",
            type="primary",
            icon=":material/search:",
        )

    if submitted:
        location = st.session_state.get("service_location", "").strip()
        service_type = st.session_state.get("service_type", "").strip()
        st.session_state.service_maps_url = None
        st.session_state.service_maps_query = None
        if not location:
            st.error("Enter a city, town, or postal code to search nearby.")
        elif not service_type:
            st.error("Enter the type of service or repair you are looking for.")
        else:
            query, url = build_maps_search_url(service_type, location)
            st.session_state.service_maps_query = query
            st.session_state.service_maps_url = url

    if st.session_state.service_maps_url:
        with st.container(border=True):
            st.markdown("**Your Google Maps search is ready**")
            st.code(st.session_state.service_maps_query, language=None)
            st.link_button(
                "View nearby shops on Google Maps",
                st.session_state.service_maps_url,
                type="primary",
                icon=":material/open_in_new:",
            )
            st.caption("Listings, opening hours, reviews, and contact details are provided by Google Maps. Confirm that a provider services your equipment before arranging a repair.")
    else:
        st.info("No service providers are listed or recommended by this app. Submit a location to open live Google Maps search results.", icon=":material/location_on:")

    st.caption("This search is separate from the failure model. The model does not identify a specific failed component or choose a repair provider.")


def render_report() -> None:
    render_header("OUTPUT / DOCUMENTATION", "Assessment Report", "Generate and download a PDF record of the current submitted assessment.")
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("A report becomes available after a successful assessment.")
        return
    st.write(f"Result: **{'Failure class predicted' if assessment['predicted_failure'] else 'Failure class not predicted'}** · {assessment['failure_probability']:.1%} · {assessment['risk_level']} risk")
    if st.button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:"):
        try:
            st.session_state.report_pdf = generate_assessment_pdf(
                assessment,
                st.session_state.assessment_values,
                st.session_state.assessment_model_filename,
                st.session_state.assessment_image_filename,
            )
        except Exception as exc:
            st.session_state.report_pdf = None
            st.error(f"Report generation failed: {exc}")
    if st.session_state.report_pdf:
        st.download_button(
            "Download assessment PDF",
            data=st.session_state.report_pdf,
            file_name="machine_health_assessment.pdf",
            mime="application/pdf",
            icon=":material/download:",
        )


def render_about() -> None:
    render_header("SYSTEM / SCOPE", "About", "A local decision-support application for industrial machine health assessment.")
    st.markdown("### Model scope")
    st.write("The app loads a previously trained Decision Tree artifact using joblib. It does not retrain the model. The expected feature schema comes from the AI4I 2020 Predictive Maintenance Dataset: five measured numeric values and machine type.")
    st.markdown("### Limitations")
    st.write("The image is only shown for operator visual reference. No sensor values are inferred from it. Model probabilities and classifications depend on the supplied artifact and its training; the app makes no independent accuracy claim. The tool is not a safety controller, diagnostic instrument, or replacement for qualified maintenance personnel.")
    st.write("The service locator sends the repair search phrase and location to Google Maps when opened. Provider listings are external and are not verified by this application.")
    st.markdown("### Model artifact")
    if model_filename := st.session_state.get("loaded_model_filename"):
        st.code(model_filename)
    else:
        st.warning("No usable model artifact is loaded. Place the trained .pkl file under models/ and restart or refresh the app.")


def apply_styles() -> None:
    st.html(
        """
        <style>
        :root {
            --ink: #202a26;
            --muted: #64716b;
            --line: #d3dad4;
            --accent: #e66d47;
            --accent-bright: #dce94f;
            --sidebar: #172923;
            --sidebar-raised: #263b33;
            --surface: #ffffff;
        }
        .stApp {
            background-color: #f0f4f0;
            background-image: linear-gradient(rgba(23, 41, 35, .035) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(23, 41, 35, .035) 1px, transparent 1px);
            background-size: 28px 28px;
            color: var(--ink);
        }
        .main .block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid #31483d; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #e5eae5; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #aebbb4; }
        .brand-lockup { display: flex; align-items: center; gap: 12px; margin: .35rem 0 .5rem; color: #f4f4ed; font-size: .82rem; line-height: 1.45; }
        .brand-mark { display: grid; place-items: center; width: 44px; height: 44px; flex: 0 0 44px; background: var(--accent); color: white; font-size: .9rem; font-weight: 800; border-radius: 5px; }
        .brand-lockup strong { color: white; font-weight: 750; }
        .sidebar-label { margin: 1.45rem 0 .45rem; color: #93a39a; font-size: .72rem; font-weight: 750; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .42rem .65rem; border-radius: 4px; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: var(--sidebar-raised); box-shadow: inset 3px 0 var(--accent-bright); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #d9e0db; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #ffffff; font-weight: 700; }
        .page-kicker { margin-bottom: .35rem; color: var(--accent); font-size: .76rem; font-weight: 750; }
        h1, h2, h3 { color: var(--ink); font-weight: 700; }
        h1 { font-size: 2rem; line-height: 1.15; }
        h2 { font-size: 1.45rem; }
        h3 { font-size: 1.08rem; }
        [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-top: 3px solid var(--accent); padding: 1rem; border-radius: 5px; }
        .step-index { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; margin-bottom: .65rem; border-radius: 4px; background: #eff3c8; color: #536018; font-size: .78rem; font-weight: 800; }
        .st-key-dashboard-hero { margin: .8rem 0 1.5rem; padding: 1.15rem 1.3rem; border: 1px solid #31483d; border-left: 5px solid var(--accent-bright); border-radius: 6px; background: #172923; }
        .st-key-dashboard-hero h1 { color: #f5f4ed; font-size: 2rem; }
        .st-key-dashboard-hero [data-testid="stMarkdownContainer"] p { color: #e0e5df; }
        .st-key-dashboard-hero [data-testid="stCaptionContainer"] p { color: #aebbb4; }
        .st-key-dashboard-hero [data-testid="stAlert"] p { color: var(--ink); }
        [data-testid="stForm"] { border-color: var(--line); border-radius: 5px; background: rgba(255, 255, 255, .92); }
        [data-testid="stFileUploaderDropzone"] { border: 1px dashed #aab7ae; border-radius: 5px; background: rgba(255, 255, 255, .65); }
        [data-testid="stAlert"] { border-radius: 5px; }
        div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] { font-weight: 700; }
        @media (max-width: 760px) {
            .main .block-container { padding: 1.2rem 1rem 3rem; }
            .st-key-dashboard-hero { padding: .8rem; }
        }
        </style>
        """
    )


def main() -> None:
    initialize_state()
    apply_styles()
    model, feature_order, model_error = load_current_model()
    model_path = find_model_file(MODELS_DIR)
    model_filename = model_path.name if model_path else None
    st.session_state.loaded_model_filename = model_filename if model is not None else None
    page = render_sidebar(model, model_filename, model_error)

    if page == "Dashboard":
        render_dashboard(model, model_filename, model_error)
    elif page == "Machine Analysis":
        render_machine_analysis(model, feature_order, model_filename, model_error)
    elif page == "Prediction Details":
        render_prediction_details(model)
    elif page == "Maintenance Recommendation":
        render_recommendations()
    elif page == "Service Locator":
        render_service_locator()
    elif page == "Assessment Report":
        render_report()
    else:
        render_about()


if __name__ == "__main__":
    main()
