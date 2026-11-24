import altair as alt
import pandas as pd
import streamlit as st

import app_core as core
import app_data
import config as cfg

res = app_data.artifacts().results
models = res["models"]
winner = res["winner"]
pers = models["Persistence"]
win = models[winner]

st.caption(f"2026 test year, {res['split']['test']:,} site-hours, scored once after the model was chosen on "
           f"2025. Thresholds: '{res['threshold_set']}', features: '{res['feature_set']}'.")

with st.container(horizontal=True):
    st.metric(f"{winner} macro-F1", f"{win['test_macro_f1']:.3f}",
              delta=f"{win['test_macro_f1'] - pers['test_macro_f1']:+.3f} vs persistence", border=True)
    st.metric(f"{winner} Critical recall", f"{win['test_crit_recall']:.1%}",
              delta=f"{win['test_crit_recall'] - pers['test_crit_recall']:+.1%} vs persistence", border=True)
    missed = res["errors"]["missed_critical"]
    critical = sum(win["cm_test"][2])  # bottom row of the confusion matrix: actually Critical
    fewer = missed - sum(pers["cm_test"][2][:2])
    st.metric("Missed Critical hours", f"{missed:,} :small[:gray[/ {critical:,}]]",
              delta=f"{fewer:+,} vs persistence", delta_color="inverse", border=True,
              help=f"Out of {critical:,} hours that were actually Critical. Forecast lower: "
                   + ", ".join(f"{v:,} as {k}" for k, v in res["errors"]["missed_predicted_as"].items())
                   + ". Hours are counted at each of the eight sites and overlap, so one storm can count many times.")
    st.metric("Prediction time", f"{win['latency_ms']:.1f} ms", border=True,
              help=f"Average time to forecast one location at one hour, on a single CPU thread. Model size {win['size_mb']:.1f} MB.")

st.subheader("All models")
table = pd.DataFrame([
    dict(Model=name + (" (chosen)" if name == winner else ""),
         **{"Val macro-F1": m["val_macro_f1"], "Val Critical recall": m["val_crit_recall"],
            "Test macro-F1": m["test_macro_f1"], "Test Critical recall": m["test_crit_recall"],
            "Size (MB)": m.get("size_mb"), "Latency (ms)": m.get("latency_ms")})
    for name, m in models.items()
])
pct = st.column_config.NumberColumn(format="percent")
st.dataframe(table, hide_index=True,
             column_config={"Val macro-F1": st.column_config.NumberColumn(format="%.3f"),
                            "Test macro-F1": st.column_config.NumberColumn(format="%.3f"),
                            "Val Critical recall": pct, "Test Critical recall": pct,
                            "Size (MB)": st.column_config.NumberColumn(format="%.2f"),
                            "Latency (ms)": st.column_config.NumberColumn(format="%.2f")},
             alt="Validation and test scores, size and latency for each model and the persistence baseline")
st.info(res["selection_rule"], title="Selection rule (validation data only)", icon=":material/rule:")

left, right = st.columns(2, gap="large")
with left:
    st.subheader(f"{winner} confusion matrix")
    cm = win["cm_test"]
    cm_df = pd.DataFrame([dict(Actual=core.CLASS_NAMES[i], Predicted=core.CLASS_NAMES[j], Hours=cm[i][j],
                               Share=cm[i][j] / sum(cm[i])) for i in range(3) for j in range(3)])
    base = alt.Chart(cm_df).encode(x=alt.X("Predicted:N", sort=core.CLASS_NAMES),
                                   y=alt.Y("Actual:N", sort=core.CLASS_NAMES))
    st.altair_chart(
        (base.mark_rect().encode(color=alt.Color("Share:Q", legend=None, scale=alt.Scale(scheme="blues")),
                                 tooltip=["Actual", "Predicted", alt.Tooltip("Hours:Q", format=","),
                                          alt.Tooltip("Share:Q", format=".1%", title="Share of actual")])
         + base.mark_text(fontSize=13).encode(
             text=alt.Text("Hours:Q", format=","),
             color=alt.condition("datum.Share > 0.5", alt.value("white"), alt.value("black"))))
        .properties(height=280),
        alt=f"{winner} test confusion matrix of actual against forecast class",
    )
with right:
    st.subheader("Critical recall by month")
    month = pd.DataFrame([dict(Month=pd.Timestamp(2026, int(m), 1).strftime("%b"), Order=int(m),
                               **{winner: v["model_crit_recall"], "Persistence": v["persist_crit_recall"]})
                          for m, v in res["errors"]["by_month"].items()])
    month = month.melt(id_vars=["Month", "Order"], var_name="Forecast", value_name="Critical recall")
    st.altair_chart(
        alt.Chart(month).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=alt.SortField("Order"), title=None),
            y=alt.Y("Critical recall:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color="Forecast:N",
            tooltip=["Month", "Forecast", alt.Tooltip("Critical recall:Q", format=".1%")],
        ).properties(height=280),
        alt=f"Critical recall by month in 2026 for {winner} and persistence",
    )

st.subheader("By location")
site_rows = []
for s, v in res["errors"]["by_site"].items():
    for metric, mk, pk in (("Macro-F1", "model_macro_f1", "persist_macro_f1"),
                           ("Critical recall", "model_crit_recall", "persist_crit_recall")):
        site_rows += [dict(Site=s, Metric=metric, Forecast=winner, Score=v[mk]),
                      dict(Site=s, Metric=metric, Forecast="Persistence", Score=v[pk])]
st.altair_chart(
    alt.Chart(pd.DataFrame(site_rows)).mark_bar().encode(
        x=alt.X("Site:N", title=None),
        xOffset="Forecast:N",
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.5, 1]), axis=alt.Axis(format="%")),
        color="Forecast:N",
        column=alt.Column("Metric:N", title=None),
        tooltip=["Site", "Metric", "Forecast", alt.Tooltip("Score:Q", format=".1%")],
    ).properties(height=260, width=380),
    alt=f"Macro-F1 and Critical recall at each site for {winner} and persistence",
)

boot = app_data.json_file(str(cfg.MODELS_DIR / "bootstrap_test.json"))
if boot:
    st.subheader("Bootstrap confidence intervals")
    names = {"base": "XGBoost, base features", "extended": "XGBoost, extended features (deployed)",
             "ext_rf": "Random Forest, extended", "ext_lr": "LogReg, extended", "persistence": "Persistence"}
    ci = pd.DataFrame([dict(Model=names.get(k, k), Metric=label, Score=v[key], Low=v[key + "_ci"][0],
                            High=v[key + "_ci"][1])
                       for k, v in boot["models"].items()
                       for key, label in (("macro_f1", "Macro-F1"), ("crit_recall", "Critical recall"))])
    order = list(ci["Model"].unique())
    pts = alt.Chart(ci).encode(y=alt.Y("Model:N", sort=order, title=None))
    st.altair_chart(
        alt.layer(
            pts.mark_rule(strokeWidth=2).encode(x=alt.X("Low:Q", scale=alt.Scale(zero=False), title="Score"),
                                                x2="High:Q"),
            pts.mark_point(filled=True, size=70).encode(
                x="Score:Q",
                tooltip=["Model", alt.Tooltip("Score:Q", format=".3f"), alt.Tooltip("Low:Q", format=".3f"),
                         alt.Tooltip("High:Q", format=".3f")]),
        ).properties(height=200, width=420).facet(column=alt.Column("Metric:N", title=None))
        .resolve_scale(x="independent"),
        alt="Test scores with 95% bootstrap intervals for each model and persistence",
    )
    diffs = pd.DataFrame([dict(Comparison=k, **{"Macro-F1 difference": v["macro_f1"],
                                                "Macro-F1 95% CI": f"{v['macro_f1_ci'][0]:+.4f} to {v['macro_f1_ci'][1]:+.4f}",
                                                "Critical recall difference": v["crit_recall"],
                                                "Critical recall 95% CI": f"{v['crit_recall_ci'][0]:+.4f} to {v['crit_recall_ci'][1]:+.4f}"})
                          for k, v in boot["differences"].items()])
    with st.expander("Paired differences"):
        st.dataframe(diffs, hide_index=True,
                     column_config={"Macro-F1 difference": st.column_config.NumberColumn(format="%+.4f"),
                                    "Critical recall difference": st.column_config.NumberColumn(format="%+.4f")},
                     alt="Paired bootstrap differences between models with 95% intervals")
    st.caption(f"{boot['bootstrap_samples']:,} resamples of whole days ({boot['days']} days) so that hours "
               "from the same weather system stay together. An interval that excludes zero is a reliable "
               "difference.")

st.subheader("Retrieval benchmark")
ret = app_data.json_file(str(cfg.RETRIEVAL_RESULTS_PATH))
if ret:
    st.dataframe(pd.DataFrame(ret["methods"]).T.rename_axis("Method").reset_index(), hide_index=True,
                 alt="Retrieval scores for BM25, dense and hybrid search")
    st.caption(f"{ret['queries_scored']} queries scored, {ret['relevant_documents']} relevant documents, "
               f"offshore documents only: {ret['core_only']}.")
else:
    st.info("Run `python src/benchmark_retrieval.py pool --core-only`, mark the pool, then "
            "`python src/benchmark_retrieval.py evaluate --core-only`. Results will appear here.",
            title="Not run yet", icon=":material/pending:")
