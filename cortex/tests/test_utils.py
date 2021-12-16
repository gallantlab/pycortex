import cortex

def test_download_subject():
    # Test that newly downloaded subjects are added to the current database.

    # remove fsaverage from the list of available subjects if present.
    if "fsaverage" in cortex.db.subjects:
        cortex.db._subjects.pop("fsaverage")

    assert "fsaverage" not in cortex.db.subjects
    cortex.utils.download_subject(subject_id='fsaverage')
    assert "fsaverage" in cortex.db.subjects
