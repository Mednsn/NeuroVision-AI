
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image


# ==========================================
# 1. Configuration de l'interface
# ==========================================

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroVision AI")
st.subheader("Classification des images IRM cérébrales")

st.write(
    "Importez une image IRM pour obtenir une prédiction "
    "parmi les quatre classes du modèle CNN."
)

st.info(
    "Projet éducatif : cette prédiction ne constitue pas "
    "un diagnostic médical."
)


# ==========================================
# 2. Charger le modèle sauvegardé
# ==========================================

MODEL_PATH = Path(__file__).parent / "neurovision_cnn.keras"


@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    if not MODEL_PATH.exists():
        st.error(
            "Modèle introuvable. Vérifiez que "
            "'neurovision_cnn.keras' est dans le dossier du projet."
        )
        st.stop()

    model = charger_modele()

except Exception as error:
    st.error(f"Erreur lors du chargement du modèle : {error}")
    st.stop()


# ==========================================
# 3. Définir les classes
# ==========================================

# Ordre utilisé par LabelEncoder.classes_
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

if model.output_shape[-1] != len(class_names):
    st.error(
        "Le nombre de classes défini dans l'interface "
        "ne correspond pas à la sortie du modèle."
    )
    st.stop()


# ==========================================
# 4. Importer une image
# ==========================================

uploaded_file = st.file_uploader(
    "Choisissez une image IRM",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 5. Prétraiter et afficher l'image
# ==========================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Redimensionner comme pendant la préparation des données
        image = image.resize((224, 224))

        # Transformer l'image en tableau NumPy
        image_array = np.array(image, dtype=np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Image IRM importée",
                use_container_width=True
            )

        with col2:
            st.write("### Informations sur l'image")
            st.write("Dimensions : 224 × 224 pixels")
            st.write("Canaux : RGB")
            st.write("Format des pixels : 0 à 255")

            lancer_prediction = st.button(
                "🔍 Analyser l'image",
                type="primary",
                use_container_width=True
            )

        # ==========================================
        # 6. Prédiction avec le CNN
        # ==========================================

        if lancer_prediction:

            with st.spinner("Analyse en cours..."):

                # Ajouter une dimension batch
                image_batch = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Obtenir les probabilités des classes
                probabilities = model.predict(
                    image_batch,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = class_names[predicted_index]
                confidence = float(
                    probabilities[predicted_index]
                )

            st.divider()
            st.header("Résultat de la prédiction")

            st.metric(
                label="Classe prédite",
                value=predicted_class
            )

            st.metric(
                label="Score du modèle",
                value=f"{confidence:.2%}"
            )

            # ==========================================
            # 7. Afficher les probabilités
            # ==========================================

            st.subheader("Probabilités par classe")

            results = pd.DataFrame({
                "Classe": class_names,
                "Probabilité": probabilities
            })

            results = results.sort_values(
                by="Probabilité",
                ascending=False
            )

            st.bar_chart(
                results.set_index("Classe")["Probabilité"]
            )

            st.dataframe(
                results.assign(
                    Probabilité=results["Probabilité"].map(
                        lambda value: f"{value:.2%}"
                    )
                ),
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "Le score représente la probabilité estimée "
                "par le modèle, et non une certitude médicale. "
                "Le résultat ne doit pas être utilisé pour "
                "établir ou exclure un diagnostic."
            )

    except Exception as error:
        st.error(
            f"Impossible de traiter cette image : {error}"
        )

else:
    st.warning(
        "Veuillez importer une image JPG, JPEG ou PNG "
        "pour commencer."
    )
