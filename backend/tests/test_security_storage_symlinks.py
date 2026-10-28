"""Storage deletion never follows symbolic links: a link is removed itself, its target stays.

Symlinks are never created by the app, but a restored .tar.gz backup can carry relative links
that stay inside the archive (tarfile's 'data' filter keeps them)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clipper.api.routes_projects import _cleanup_work_dir
from clipper.storage import Storage

P_X = "p_0123456789abcdef"
P_A = "p_aaaaaaaaaaaaaaaa"
C_B = "c_bbbbbbbbbbbbbbbb"
C_C = "c_cccccccccccccccc"


def _other_clip(storage: Storage) -> Path:
    d = storage.clip_dir(P_A, C_B)
    d.mkdir(parents=True)
    (d / "render.mp4").write_bytes(b"other project's clip")
    return d


def test_cleanup_work_dir_removes_link_not_database(storage: Storage) -> None:
    db_file = storage.root / "clipper.db"
    db_file.write_bytes(b"SQLite format 3\x00")
    work = storage.work_dir(P_X)
    work.mkdir(parents=True)
    link = work / "chunks.json"
    os.symlink("../../../clipper.db", link)
    assert link.resolve() == db_file
    _cleanup_work_dir(storage, P_X)
    assert db_file.read_bytes() == b"SQLite format 3\x00"
    assert not link.is_symlink() and not link.exists()


def test_delete_file_removes_link_only(storage: Storage) -> None:
    target = storage.root / "clipper.db"
    target.write_bytes(b"db")
    link = storage.root / "projects" / "link.json"
    os.symlink(target, link)
    storage.delete_file(link)
    assert target.read_bytes() == b"db" and not os.path.lexists(link)


def test_delete_tree_on_symlinked_clip_dir_keeps_other_project(storage: Storage) -> None:
    other = _other_clip(storage)
    link = storage.clip_dir(P_X, C_C)
    link.parent.mkdir(parents=True)
    os.symlink(f"../../{P_A}/clips/{C_B}", link)
    storage.delete_tree(link)
    assert (other / "render.mp4").read_bytes() == b"other project's clip"
    assert not os.path.lexists(link)


def test_delete_tree_does_not_descend_into_links(storage: Storage) -> None:
    other = _other_clip(storage)
    project = storage.project_dir(P_X)
    (project / "clips").mkdir(parents=True)
    os.symlink(other, project / "clips" / C_C)
    os.symlink(other / "render.mp4", project / "stolen.mp4")
    storage.delete_tree(project)
    assert not project.exists() and (other / "render.mp4").exists()


def test_delete_file_refuses_path_through_symlinked_directory(storage: Storage) -> None:
    other = _other_clip(storage)
    link = storage.clip_dir(P_X, C_C)
    link.parent.mkdir(parents=True)
    os.symlink(f"../../{P_A}/clips/{C_B}", link)
    with pytest.raises(ValueError):
        storage.delete_file(link / "render.mp4")
    with pytest.raises(ValueError):
        storage.delete_tree(link / "render.mp4")
    assert (other / "render.mp4").exists()


def test_link_to_allowed_area_cannot_stand_in_for_it(storage: Storage, tmp_path: Path) -> None:
    """A link whose own location is outside the managed folders is refused, wherever it points."""
    d = storage.project_dir(P_X)
    d.mkdir(parents=True)
    outside_link = tmp_path / "elsewhere"
    os.symlink(d, outside_link)
    with pytest.raises(ValueError):
        storage.delete_tree(outside_link)
    assert d.is_dir() and outside_link.is_symlink()


def test_data_dir_reached_through_a_symlink_still_works(tmp_path: Path) -> None:
    """A data dir configured via a symlinked path (e.g. /var -> /private/var) keeps working."""
    real = tmp_path / "real-data"
    real.mkdir()
    alias = tmp_path / "alias-data"
    os.symlink(real, alias)
    st = Storage(alias)
    st.ensure()
    via_alias = alias / "projects" / P_X
    via_alias.mkdir(parents=True)
    (via_alias / "a.txt").write_text("x")
    st.delete_file(via_alias / "a.txt")
    assert not (real / "projects" / P_X / "a.txt").exists()
    st.delete_tree(via_alias)
    assert not (real / "projects" / P_X).exists()
