import shutil
import tarfile
from unittest import mock

import pytest

import cortex


@pytest.fixture
def fake_fsaverage_tarball(tmp_path):
    """Build a minimal fsaverage.tar.gz that pycortex will recognize as a subject."""
    subj_src = tmp_path / "src" / "fsaverage"
    subj_src.mkdir(parents=True)
    (subj_src / "marker").write_text("ok")
    tarball = tmp_path / "fsaverage.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(subj_src, arcname="fsaverage")
    return tarball


@pytest.fixture
def isolated_filestore(tmp_path, monkeypatch):
    """Redirect cortex.db.filestore to an empty tmp dir and reset the subject cache."""
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(cortex.db, "filestore", str(store))
    original_subjects = cortex.db._subjects
    cortex.db._subjects = None
    yield store
    cortex.db._subjects = original_subjects


def test_download_subject(isolated_filestore, fake_fsaverage_tarball, monkeypatch):
    # Newly downloaded subjects are added to the current database.
    def fake_retrieve(url, dest):
        shutil.copy(fake_fsaverage_tarball, dest)
        return dest, None

    mock_retrieve = mock.Mock(side_effect=fake_retrieve)
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    assert "fsaverage" not in cortex.db.subjects
    cortex.utils.download_subject(subject_id='fsaverage')
    assert "fsaverage" in cortex.db.subjects
    assert mock_retrieve.call_count == 1


def test_download_subject_skips_when_present(isolated_filestore, monkeypatch):
    # If the subject is already in the database and download_again is False,
    # download_subject warns and returns without touching the network.
    cortex.db._subjects = {"fsaverage": mock.MagicMock()}

    mock_retrieve = mock.Mock()
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    with pytest.warns(UserWarning, match="already present"):
        cortex.utils.download_subject(subject_id='fsaverage')
    mock_retrieve.assert_not_called()


def test_download_subject_download_again(
    isolated_filestore, fake_fsaverage_tarball, monkeypatch
):
    # download_again=True re-downloads even when the subject is already present.
    def fake_retrieve(url, dest):
        shutil.copy(fake_fsaverage_tarball, dest)
        return dest, None

    mock_retrieve = mock.Mock(side_effect=fake_retrieve)
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    cortex.utils.download_subject(subject_id='fsaverage')
    assert "fsaverage" in cortex.db.subjects
    cortex.utils.download_subject(subject_id='fsaverage', download_again=True)
    assert mock_retrieve.call_count == 2
