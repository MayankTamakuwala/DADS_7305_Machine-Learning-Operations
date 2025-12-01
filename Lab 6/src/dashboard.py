import pandas as pd
import streamlit as st

from main import train_and_evaluate, save_model


st.set_page_config(
    page_title="Breast Cancer Trainer",
    page_icon="🧠",
    layout="wide",
)
st.title("Breast Cancer Model Trainer")
st.write(
    "Tweak logistic regression hyperparameters and kick off training. "
    "Results and diagnostics appear once the run completes."
)


with st.sidebar:
    st.header("Hyperparameters")
    c = st.select_slider(
        "Inverse regularization strength (C)",
        options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        value=1.0,
    )
    max_iter = st.slider("Max iterations", min_value=100, max_value=1000, value=500, step=50)
    test_size = st.slider("Test split size", min_value=0.1, max_value=0.4, value=0.25, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)

    st.header("Outputs")
    save_artifact = st.checkbox("Save trained model", value=True)
    artifact_path = st.text_input("Artifact path", value="artifacts/breast_cancer_dashboard.pkl")

    train_clicked = st.button("Train model", type="primary")


def render_metrics(metrics, target_names):
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    st.subheader("Classification report")
    report_df = pd.DataFrame(metrics["classification_report_dict"]).T
    st.table(report_df)

    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=pd.Index(target_names, name="True"),
        columns=pd.Index(target_names, name="Predicted"),
    )
    st.table(cm_df)


if train_clicked:
    with st.spinner("Training model..."):
        model, metrics, feature_names, target_names = train_and_evaluate(
            random_state=int(random_state),
            test_size=float(test_size),
            c=float(c),
            max_iter=int(max_iter),
        )

        saved_path = None
        if save_artifact:
            target_path = artifact_path.strip() or "artifacts/breast_cancer_dashboard.pkl"
            saved_path = save_model(model, feature_names, target_names, model_path=target_path)

    st.success("Training complete.")
    if saved_path:
        st.caption(f"Model persisted to {saved_path}")

    render_metrics(metrics, target_names)
else:
    st.info("Configure parameters on the left, then click 'Train model' to begin.")
