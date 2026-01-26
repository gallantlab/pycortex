"""Tests for cortex.segment module, specifically for issue #591."""

import pytest
from unittest.mock import patch
import numpy as np

from cortex import segment


class TestCutSurfaceFreesurferSubjectDir:
    """Test cases for freesurfer_subject_dir parameter handling in cut_surface()."""

    @pytest.fixture
    def mock_get_surf_return(self):
        """Return value for mocked freesurfer.get_surf calls."""
        # Return dummy points, polygons, and normals with matching vertex counts
        pts = np.zeros((100, 3))
        polys = np.zeros((50, 3), dtype=int)
        norms = np.zeros((100, 3))
        return pts, polys, norms

    def test_freesurfer_subject_dir_passed_to_get_surf(
        self, mock_get_surf_return, monkeypatch
    ):
        """Regression test for issue #591.

        Ensure freesurfer_subject_dir is passed to get_surf() calls.
        Without the fix, this test would fail when SUBJECTS_DIR is not set.
        """
        # Remove SUBJECTS_DIR to ensure test doesn't rely on environment
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)

        custom_subjects_dir = "/custom/freesurfer/subjects"

        with patch("cortex.segment.freesurfer.get_surf") as mock_get_surf:
            mock_get_surf.return_value = mock_get_surf_return

            # Call cut_surface with explicit freesurfer_subject_dir
            # We expect it to fail later (e.g., at db.get_paths), but we want to
            # verify that get_surf is called correctly first
            try:
                segment.cut_surface(
                    cx_subject="test_subject",
                    hemi="lh",
                    freesurfer_subject_dir=custom_subjects_dir,
                )
            except Exception:
                # We expect failures after get_surf (e.g., database access)
                pass

            # Verify get_surf was called twice (for inflated and fiducial)
            assert mock_get_surf.call_count == 2

            # Verify freesurfer_subject_dir was passed to both calls
            for call in mock_get_surf.call_args_list:
                assert call.kwargs.get("freesurfer_subject_dir") == custom_subjects_dir

    def test_raises_keyerror_without_subjects_dir(self, monkeypatch):
        """Test that KeyError is raised when SUBJECTS_DIR is not set and no dir provided.

        This tests the expected behavior when neither SUBJECTS_DIR environment
        variable nor freesurfer_subject_dir parameter is provided.
        """
        # Remove SUBJECTS_DIR environment variable
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)

        # Without the freesurfer_subject_dir parameter and without SUBJECTS_DIR,
        # the function should eventually raise a KeyError from freesurfer.get_surf
        with pytest.raises(KeyError):
            segment.cut_surface(
                cx_subject="test_subject",
                hemi="lh",
                # freesurfer_subject_dir not provided
            )

    def test_backward_compatibility_with_subjects_dir_env(
        self, mock_get_surf_return, monkeypatch
    ):
        """Test backward compatibility when SUBJECTS_DIR env var is set.

        When freesurfer_subject_dir is not provided, the function should still
        work if SUBJECTS_DIR environment variable is set (None is passed to
        get_surf which handles the fallback internally).
        """
        # Set SUBJECTS_DIR environment variable
        monkeypatch.setenv("SUBJECTS_DIR", "/env/freesurfer/subjects")

        with patch("cortex.segment.freesurfer.get_surf") as mock_get_surf:
            mock_get_surf.return_value = mock_get_surf_return

            try:
                segment.cut_surface(
                    cx_subject="test_subject",
                    hemi="lh",
                    # freesurfer_subject_dir not provided - should use env var via None
                )
            except Exception:
                # We expect failures after get_surf (e.g., database access)
                pass

            # Verify get_surf was called twice
            assert mock_get_surf.call_count == 2

            # Verify freesurfer_subject_dir=None was passed (internal fallback)
            for call in mock_get_surf.call_args_list:
                assert call.kwargs.get("freesurfer_subject_dir") is None
