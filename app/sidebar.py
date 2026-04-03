"""
app/sidebar.py
--------------
Sidebar rendering and sidebar-triggered actions for the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.ui_helpers import build_material_source, get_profile
from project_paths import PROFILES_DIR
from pipeline.material_ingestion import (
    delete_all_materials_everywhere,
    delete_material_everywhere,
)
from storage.materials_db import list_materials


@dataclass
class SidebarState:
    user_id: str
    materials: list[dict]
    material_options: dict[str, str | None]
    material_sel: str
    diff_sel: str
    use_reranker: bool
    show_timings: bool


def _render_profile_section(user_id: str) -> None:
    profile = get_profile(user_id)
    st.subheader("Your Learning Profile")
    difficulty_label = profile.preferred_difficulty.capitalize()
    colours = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.markdown(f"**Current level:** {colours.get(difficulty_label, '⚪')} {difficulty_label}")

    top = profile.top_topics
    st.markdown(f"**Top topics:** {', '.join(top) if top else 'None yet — start asking!'}")

    recs = profile.recommended_topics
    if recs:
        st.markdown(f"**Suggested to explore:** {', '.join(recs)}")


def _render_materials_section(materials: list[dict]) -> None:
    st.divider()
    st.subheader("Stored Materials")

    if not materials:
        st.caption("No course PDFs stored yet.")
        return

    st.caption(f"{len(materials)} PDF(s) stored in the database")
    selected_materials: list[dict] = []

    if st.button("Delete All Materials", key="delete_all_materials"):
        with st.spinner("Deleting all materials from the database and Qdrant..."):
            result = delete_all_materials_everywhere()
        st.session_state["materials_notice"] = {
            "level": "success" if result["deleted"] else "error",
            "message": result["message"],
        }
        st.rerun()

    for material in materials[:6]:
        checked = st.checkbox(
            material["filename"],
            key=f"select_material_{material['id']}",
        )
        if checked:
            selected_materials.append(material)

    if st.button("Delete Selected", key="delete_selected_materials"):
        if not selected_materials:
            st.session_state["materials_notice"] = {
                "level": "warning",
                "message": "Select at least one material to delete.",
            }
            st.rerun()

        deleted_names: list[str] = []
        failed_messages: list[str] = []
        with st.spinner("Deleting selected materials from the database and Qdrant..."):
            for material in selected_materials:
                result = delete_material_everywhere(material["id"])
                if result["deleted"]:
                    deleted_names.append(material["filename"])
                else:
                    failed_messages.append(result["message"])
                st.session_state.pop(f"select_material_{material['id']}", None)

        if failed_messages:
            message_parts = []
            if deleted_names:
                message_parts.append(
                    f"Deleted {len(deleted_names)} material(s): {', '.join(deleted_names)}."
                )
            message_parts.extend(failed_messages)
            st.session_state["materials_notice"] = {
                "level": "error",
                "message": " ".join(message_parts),
            }
        else:
            st.session_state["materials_notice"] = {
                "level": "success" if deleted_names else "warning",
                "message": (
                    f"Deleted {len(deleted_names)} material(s): {', '.join(deleted_names)}."
                    if deleted_names
                    else "No selected materials were deleted."
                ),
            }
        st.rerun()


def render_sidebar() -> SidebarState:
    with st.sidebar:
        st.title("📚 NLP Assistant")
        st.divider()

        materials_notice = st.session_state.pop("materials_notice", None)
        if materials_notice:
            getattr(st, materials_notice["level"])(materials_notice["message"])

        user_id = st.text_input("Your student ID", value="student_1", key="user_id_input")

        _render_profile_section(user_id)

        materials = list_materials()
        _render_materials_section(materials)

        st.divider()
        st.subheader("Retrieval Filters")
        st.caption("Optionally restrict answers by stored course material or difficulty.")

        material_options = {"Any": None}
        for material in materials:
            material_options[material["filename"]] = build_material_source(material)

        material_sel = st.selectbox("Course Material", list(material_options.keys()))
        diff_sel = st.selectbox("Difficulty", ["Any", "beginner", "intermediate", "advanced"])
        use_reranker = st.toggle("Enable reranker (more accurate, slower)", value=True)
        show_timings = st.toggle("Show latency breakdown", value=False)

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset my learning profile"):
            profile_path = PROFILES_DIR / f"{user_id}.json"
            if profile_path.exists():
                profile_path.unlink()
            st.session_state["profile_stale"] = True
            st.success("Profile reset.")
            st.rerun()

    return SidebarState(
        user_id=user_id,
        materials=materials,
        material_options=material_options,
        material_sel=material_sel,
        diff_sel=diff_sel,
        use_reranker=use_reranker,
        show_timings=show_timings,
    )
