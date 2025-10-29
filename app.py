# app.py
from __future__ import annotations
import json
import streamlit as st
import streamlit.components.v1 as components

from config import SCHEMA_VERSION
from caching import cache_views, resource_build_map_html

st.set_page_config(page_title="Analyse TripModifications + GTFS — JSON/PB/Textproto + carte", layout="wide")
st.title("Analyse TripModifications (JSON/PB/Textproto) vs GTFS — Carte Folium")
st.caption(
    "Détour (rouge), tracé originel (vert, issu de shapes.txt), "
    "arrêts originels (blancs, avec départ en vert et terminus en rouge), "
    "arrêts de remplacement (rose, positionnés avec stop_lat/stop_lon si fournis). "
    "Segments **ajoutés (turquoise)** et **annulés (violet)** s'affichent lorsque fournis. "
    "Polylines nettoyées, analyse et diagnostics pré‑calculés et mis en cache. "
    "Carte HTML pré‑rendue et centrée strictement sur le détour."
)

with st.sidebar:
    st.header("Données d’entrée")
    with st.form("inputs_form"):
        tripmods_file = st.file_uploader("TripModifications (JSON / PB / textproto)", type=["json", "pb", "pbf", "bin", "txt"], key="tripmods_up")
        gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"], key="gtfs_up")
        decode_mode = st.selectbox("Décodage polylines", ["Auto (recommandé)", "Précision 1e-5", "Précision 1e-6"], index=0, key="decode_sel")
        dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False, key="dump_first_cb")
        submitted = st.form_submit_button("Analyser", type="primary")
decode_flag = {"Auto (recommandé)": "auto", "Précision 1e-5": "p5", "Précision 1e-6": "p6"}[st.session_state.get("decode_sel", "Auto (recommandé)")]

if submitted:
    if not tripmods_file or not gtfs_file:
        st.error("Merci de sélectionner **TripModifications** (JSON/PB/textproto) **et** **GTFS** (.zip).")
    else:
        tripmods_bytes = tripmods_file.getvalue()
        gtfs_bytes = gtfs_file.getvalue()
        res = cache_views(tripmods_bytes, gtfs_bytes, decode_flag, SCHEMA_VERSION)
        st.session_state["last_results"] = res
        st.session_state["last_params"] = dict(decode_flag=decode_flag, schema_version=SCHEMA_VERSION)
        st.success("Analyse terminée ✅")

# Récupération des résultats (et vérif de version)
res = st.session_state.get("last_results")
if res and res.get("schema_version") != SCHEMA_VERSION:
    res = None
    st.session_state.pop("last_results", None)

if not res:
    st.info("Charge un TripModifications (JSON / PB / textproto) puis un GTFS (.zip), choisis la précision de décodage, et clique **Analyser**.")
    st.caption("Astuce : si ta polyligne vient d’un JSON, elle doit être « déséchappée » (retrait des \\\\ et \\n).")
    st.stop()

feed_json = res["feed_json"]
reports_plain = res["reports"]; totals = res["totals"]; total_shapes = res["total_shapes"]
needed_trip_ids = res["needed_trip_ids"]; needed_stop_ids = res["needed_stop_ids"]
details_tables_by_entity = res["details_tables_by_entity"]
temp_stops_points_by_entity = res["temp_stops_points_by_entity"]
shape_for_plot_by_entity = res["shape_for_plot_by_entity"]
shapes_plain = res["shapes_plain"]
original_poly_by_entity = res["original_poly_by_entity"]
original_shape_id_by_entity = res["original_shape_id_by_entity"]
original_stop_points_by_entity = res["original_stop_points_by_entity"]
original_stop_ids_by_entity = res["original_stop_ids_by_entity"]
added_segments_by_entity = res["added_segments_by_entity"]
canceled_segments_by_entity = res["canceled_segments_by_entity"]
gtfs_kpi = res["gtfs_kpi"]

# KPIs
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Entités", totals["total_entities"])
c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
c3.metric("modifications", totals["total_modifications"])
c4.metric("trip_ids manquants", totals["missing_trip_ids"])
c5.metric("selectors non résolus", totals["invalid_selectors"])
c6.metric("repl. stops inconnus GTFS", totals["unknown_replacement_stops"])
c7.metric("Shapes dans le feed (RT)", total_shapes)

# Liste des trip_id manquants
if totals.get("missing_trip_ids", 0) > 0:
    missing = res.get("missing_trip_ids", [])
    if missing:
        with st.expander("🚫 Voir la liste des trip_id manquants (uniques)"):
            st.write(f"**{len(missing)}** trip_id absents du GTFS (uniques).")
            st.code("\n".join(missing), language="text")
            st.download_button("⬇️ Télécharger la liste (TXT)", data="\n".join(missing),
                               file_name="trip_ids_manquants.txt", mime="text/plain")

st.info(f"Filtrage GTFS → trips requis: {len(needed_trip_ids):,} · stops requis: {len(needed_stop_ids):,} · shapes RT disponibles: {total_shapes:,}")
st.success(f"GTFS filtré : **{gtfs_kpi['trips']:,} trips**, **{gtfs_kpi['stop_times']:,} stop_times**, **{gtfs_kpi['stops_present']:,} stops**")
st.success(f"TripModifications : **{len(reports_plain)} entités**")

if st.session_state.get("dump_first_cb") and reports_plain:
    with st.expander("Aperçu du 1er trip_mod (normalisé)"):
        st.json(reports_plain[0])
if feed_json is not None:
    with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
        st.json(feed_json)

# Synthèse par entité
table = [{
    "entity_id": r["entity_id"],
    "trip_ids (sélectionnés)": r["total_selected_trip_ids"],
    "modifications": r["modification_count"],
    "service_dates": ", ".join(r["service_dates"]),
    "repl_stops inconnus": ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else ""
} for r in reports_plain if r.get("modification_count", 0) > 0]
st.subheader("Synthèse par entité")
st.dataframe(table, width="stretch", height=360)

# Détails par entité
st.subheader("Détails")
for r in reports_plain[:200]:
    ent_id = r["entity_id"]
    if r["modification_count"] <= 0:
        continue
    with st.expander(f"Entité {ent_id} — {r['total_selected_trip_ids']} trips — {r['modification_count']} modifications"):
        st.write("**Dates** :", ", ".join(r["service_dates"]) if r["service_dates"] else "—")
        st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :", ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else "—")

        rows = details_tables_by_entity.get(ent_id, [])
        st.dataframe(rows, width="stretch", height=260) if rows else st.info("Aucune ligne de diagnostic pour cette entité.")

        rt_shape_id_for_plot = shape_for_plot_by_entity.get(ent_id)
        orig_shape_id = original_shape_id_by_entity.get(ent_id)
        st.write(f"**Shape détour (RT)** : {rt_shape_id_for_plot or '—'}")
        st.write(f"**Shape originel (GTFS shapes.txt)** : {orig_shape_id or '—'}")

        if rt_shape_id_for_plot:
            coords_list = shapes_plain.get(rt_shape_id_for_plot, [])
            poly = [(float(la), float(lo)) for la, lo in coords_list]
            if poly and len(poly) >= 2:
                poly_key = tuple((round(la, 6), round(lo, 6)) for la, lo in poly)
                tmp_pts = temp_stops_points_by_entity.get(ent_id, [])
                stops_key = tuple((round(p[0], 6), round(p[1], 6), str(p[2])) for p in tmp_pts)
                orig_list = original_poly_by_entity.get(ent_id, [])
                orig_poly = [(float(la), float(lo)) for la, lo in orig_list]
                original_poly_key = tuple((round(la, 6), round(lo, 6)) for la, lo in orig_poly) if orig_poly else tuple()
                orig_stop_pts = original_stop_points_by_entity.get(ent_id, [])
                orig_stops_key = tuple((round(p[0], 6), round(p[1], 6), str(p[2])) for p in orig_stop_pts)

                add_list = added_segments_by_entity.get(ent_id, [])
                can_list = canceled_segments_by_entity.get(ent_id, [])
                def _segkey(lst):
                    return tuple(
                        tuple((round(p[0], 6), round(p[1], 6)) for p in seg if isinstance(p, list) and len(p) == 2)
                        for seg in lst
                    )
                added_key = _segkey(add_list)
                canceled_key = _segkey(can_list)

                map_html = resource_build_map_html(
                    rt_shape_id_for_plot or "",
                    poly_key,
                    stops_key,
                    original_poly_key,
                    orig_stops_key,
                    orig_shape_id or "",
                    added_key,
                    canceled_key,
                    SCHEMA_VERSION
                )
                components.html(map_html, height=460, scrolling=False)
                if orig_stop_pts:
                    st.markdown("**Arrêts du tracé originel (ordre `stop_times`) :**")
                    ids_list = original_stop_ids_by_entity.get(ent_id, [])
                    st.markdown("\n".join(f"{i+1}. `{sid}`" for i, sid in enumerate(ids_list)))
            else:
                st.info("Polyline (détour) vide ou invalide pour cette entité.")
        else:
            st.info("Aucune polyline 'encoded_polyline' (détour RT) disponible pour cette entité.")

# Export JSON
export_json = {
    "schema_version": SCHEMA_VERSION,
    "totals": totals,
    "total_shapes": total_shapes,
    "entities": reports_plain,
}
st.download_button("📥 Télécharger le rapport JSON",
    data=json.dumps(export_json, ensure_ascii=False, indent=2),
    file_name="rapport_tripmods.json", mime="application/json"
)