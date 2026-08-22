"""Tests for the database access interfaces (attribute and item access)."""
import json
import os

import numpy as np
import pytest

from cortex.database import Database

SUBJECT = "S1-test"


@pytest.fixture
def filestore(tmp_path):
    """A minimal filestore containing a subject whose name is not an identifier."""
    subj = tmp_path / SUBJECT
    for dirname in ["transforms", "anatomicals", "cache", "surfaces", "surface-info", "views"]:
        (subj / dirname).mkdir(parents=True)
    for hemi in ["lh", "rh"]:
        (subj / "surfaces" / f"flat-orig_{hemi}.gii").write_text("")
    xfmdir = subj / "transforms" / "full-head"
    xfmdir.mkdir()
    (xfmdir / "matrices.xfm").write_text(json.dumps({
        "subject": SUBJECT,
        "coord": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }))
    return str(tmp_path)


@pytest.fixture
def db(filestore):
    return Database(filestore)


def test_getitem_subject(db):
    # Subject names that are not valid python identifiers are reachable by item access.
    assert db[SUBJECT].subject == SUBJECT


def test_getattr_still_works(filestore):
    os.makedirs(os.path.join(filestore, "S1", "surfaces"))
    os.makedirs(os.path.join(filestore, "S1", "transforms"))
    db = Database(filestore)
    assert db.S1.subject == "S1"


def test_getitem_missing_subject_lists_available(db):
    with pytest.raises(KeyError) as excinfo:
        db["nosuchsubject"]
    assert SUBJECT in str(excinfo.value)


def test_getattr_missing_subject_points_at_item_access(db):
    with pytest.raises(AttributeError) as excinfo:
        db.nosuchsubject
    assert "db['nosuchsubject']" in str(excinfo.value)


def test_contains_and_iter(db):
    assert SUBJECT in db
    assert "nosuchsubject" not in db
    assert list(db) == [SUBJECT]


def test_dunder_lookup_does_not_hit_filestore(db):
    # Dunders must not be resolved against the filestore, or copy/pickle recurse.
    with pytest.raises(AttributeError):
        db.__deepcopy__
    assert db._subjects is None


def test_surfaces_getitem(db):
    surfaces = db[SUBJECT].surfaces
    # 'flat-orig' is not a valid identifier, but item access finds it.
    assert surfaces["flat-orig"].surftype == "flat-orig"
    with pytest.raises(KeyError) as excinfo:
        surfaces["nosuchsurface"]
    assert "flat-orig" in str(excinfo.value)


def test_transforms_getitem_and_getattr(db):
    transforms = db[SUBJECT].transforms
    assert transforms["full-head"].name == "full-head"
    with pytest.raises(KeyError) as excinfo:
        transforms["nosuchxfm"]
    assert "full-head" in str(excinfo.value)
    with pytest.raises(AttributeError):
        transforms.nosuchxfm


def test_xfmset_getitem(db):
    xfmset = db[SUBJECT].transforms["full-head"]
    assert np.allclose(np.asarray(xfmset["coord"].xfm), np.eye(4))
    with pytest.raises(KeyError) as excinfo:
        xfmset["magnet"]
    assert "coord" in str(excinfo.value)


def test_warning_is_raised_on_item_access(db, filestore):
    with open(os.path.join(filestore, SUBJECT, "warning.txt"), "w") as fp:
        fp.write("this subject is a test")
    db.reload_subjects()
    with pytest.warns(UserWarning, match="this subject is a test"):
        db[SUBJECT]
