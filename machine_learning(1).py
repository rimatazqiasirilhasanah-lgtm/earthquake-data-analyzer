"""Fitur tambahan Decision Tree untuk Earthquake Data Analyzer."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


def prepare_data(df, mmi_column=None):
    if mmi_column is None:
        # Label ditentukan oleh aturan magnitudo yang sudah ada di app.py.
        # Magnitudo dan energi sengaja tidak digunakan sebagai fitur.
        features = ["depth", "jarak_bdg"]
        data = df[features + ["kelas"]].copy().rename(columns={"kelas": "target"})
    else:
        # Kolom MMI harus merupakan hasil pengamatan untuk lokasi referensi sama.
        features = ["mag", "depth", "jarak_bdg"]
        data = df[features + [mmi_column]].copy().rename(columns={mmi_column: "target"})
        roman = dict(zip("I II III IV V VI VII VIII IX X XI XII".split(), range(1, 13)))
        values = data["target"].astype(str).str.strip().str.upper()
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric.fillna(values.map(roman))
        valid = numeric.between(1, 12) & (numeric % 1 == 0)
        data = data.loc[valid].copy()
        data["target"] = numeric.loc[valid].astype(int).astype(str)
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.replace([float("inf"), -float("inf")], float("nan")).dropna()
    return data.reset_index(drop=True), features


def train_model(data, features, mmi=False):
    counts = data["target"].value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError("Perlu minimal dua kategori dan setidaknya 5 data pada setiap kategori untuk pembagian 80:20. Lebih banyak data sangat disarankan.")
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["target"], test_size=0.20,
        random_state=42, stratify=data["target"]
    )
    if set(y_test) != set(y_train):
        raise ValueError("Tidak semua kategori terwakili pada data uji dan latih.")
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    labels = sorted(counts.index, key=int) if mmi else [c for c in ["Rendah", "Sedang", "Kuat"] if c in counts.index]
    return dict(model=model, X_train=X_train, X_test=X_test, y_test=y_test,
                predictions=predictions, labels=labels,
                baseline_accuracy=accuracy_score(y_test, baseline.predict(X_test)))


def tampilkan_machine_learning(df):
    import streamlit as st

    st.divider()
    st.header("8. Machine Learning – Decision Tree")
    st.caption("Supervised learning | Data latih 80% dan data uji 20% | Stratified split")
    mode = st.radio("Target klasifikasi", ["Eksperimen kategori magnitudo", "Intensitas MMI teramati"], key="ml_mode")
    mmi_column = None
    mmi = mode == "Intensitas MMI teramati"
    if mmi:
        excluded = {"waktu", "lat", "lon", "mag", "depth", "jarak_bdg", "kelas", "wib", "energi_J"}
        options = [c for c in df.columns if c not in excluded]
        if not options:
            st.warning("Dataset belum memiliki kolom MMI teramati. Tambahkan data intensitas hasil pengamatan pada lokasi Bandung yang konsisten.")
            return
        selection = st.selectbox("Pilih kolom MMI hasil pengamatan di lokasi referensi Bandung", [None] + options, key="ml_mmi", format_func=lambda x: "-- Pilih kolom --" if x is None else str(x))
        if selection is None:
            return
        mmi_column = selection
        if not st.checkbox("Saya memastikan kolom ini berisi MMI hasil pengamatan pada lokasi referensi Bandung yang konsisten, bukan magnitudo atau MMI hasil rumus.", key="ml_confirm_mmi"):
            return
        st.info("MMI harus berupa angka 1–12 atau angka Romawi I–XII. Nilai lain tidak digunakan.")
    else:
        st.info("Eksperimen: model mencoba mengestimasi kategori magnitudo dari kedalaman dan jarak. Label berasal dari aturan magnitudo aplikasi, bukan intensitas atau dampak teramati. Magnitudo dan energi tidak dipakai sebagai fitur untuk mencegah kebocoran target. Bila magnitudo sudah diketahui, aturan langsung lebih tepat.")
    data, features = prepare_data(df, mmi_column)
    st.write("Jumlah data layak untuk model:", len(data))
    st.write("Fitur masukan:", ", ".join(features))
    st.dataframe(data["target"].value_counts().rename("Jumlah").to_frame())
    try:
        result = train_model(data, features, mmi)
    except ValueError as exc:
        st.warning(str(exc))
        return
    model, X_test = result["model"], result["X_test"]
    y_test, predictions, labels = result["y_test"], result["predictions"], result["labels"]
    a, b, c = st.columns(3)
    a.metric("Akurasi data uji", f"{accuracy_score(y_test, predictions):.1%}")
    b.metric("F1-score macro", f"{f1_score(y_test, predictions, average='macro', zero_division=0):.3f}")
    c.metric("Akurasi baseline mayoritas", f"{result['baseline_accuracy']:.1%}")
    st.write(f"Data latih: {len(result['X_train'])} | Data uji: {len(X_test)}")
    st.subheader("Classification Report")
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3))
    st.subheader("Confusion Matrix")
    st.caption("Baris = kelas aktual; kolom = kelas prediksi.")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels))
    st.subheader("Perbandingan pada Data Uji")
    comparison = X_test.copy()
    comparison["aktual"] = y_test
    comparison["prediksi"] = predictions
    comparison["benar"] = comparison["aktual"] == comparison["prediksi"]
    st.dataframe(comparison, use_container_width=True)
    st.download_button("Download Hasil Pengujian ML", comparison.to_csv(index=False).encode("utf-8"), "hasil_pengujian_ml.csv", "text/csv", key="ml_download")
    st.subheader("Pengaruh Fitur pada Model")
    importance = pd.Series(model.feature_importances_, index=features, name="Importance")
    st.bar_chart(importance)
    st.caption("Feature importance menunjukkan penggunaan fitur dalam pohon, bukan hubungan sebab-akibat.")
    with st.expander("Lihat struktur Decision Tree"):
        st.code(export_text(model, feature_names=features), language="text")
    st.subheader("Uji Prediksi Data Baru")
    with st.form("ml_prediction"):
        values = {}
        feature_labels = {"mag": "Magnitudo", "depth": "Kedalaman (km)", "jarak_bdg": "Jarak episentrum ke Bandung (km)"}
        for feature in features:
            lower, upper = float(result["X_train"][feature].min()), float(result["X_train"][feature].max())
            values[feature] = st.number_input(feature_labels[feature], min_value=0.0, value=float(result["X_train"][feature].median()), key=f"ml_input_{feature}", help=f"Rentang data latih: {lower:.2f}–{upper:.2f}")
        submitted = st.form_submit_button("Prediksi Kategori")
    if submitted:
        outside = [f for f in features if not result["X_train"][f].min() <= values[f] <= result["X_train"][f].max()]
        if outside:
            st.warning("Masukan di luar rentang data latih. Prediksi tidak ditampilkan.")
        else:
            new_data = pd.DataFrame([values], columns=features)
            predicted = model.predict(new_data)[0]
            st.success(f"Hasil estimasi model: {'MMI ' if mmi else ''}{predicted}")
    st.caption("Evaluasi awal menggunakan pembagian acak, bukan validasi temporal/spasial. Model tidak memprediksi kapan atau di mana gempa akan terjadi dan bukan sistem peringatan dini.")
