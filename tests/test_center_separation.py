import numpy as np
import pytest
from scipy import ndimage

from src import center_separation as cs
from src import utils


def disk(shape, center, radius):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


def test_overlapping_disks_are_separated():
    img = (disk((60, 80), (30, 25), 15) | disk((60, 80), (30, 52), 15)).astype(np.uint8)
    centers, center_distances, relabeled = cs.separate_objects(img, 15, centers_only=False)

    assert centers.max() == 2
    assert np.array_equal(relabeled > 0, img > 0)
    assert relabeled[30, 25] != relabeled[30, 52]
    assert relabeled[30, 25] == centers[np.nonzero(centers == relabeled[30, 25])].item()
    assert center_distances[centers > 0].min() >= 14


def test_small_instance_next_to_large_one_gets_its_own_center():
    img = disk((60, 60), (30, 25), 15).astype(np.uint8)
    img[28:31, 43:46] = 1  # A small separate blob, within the max filter range of the large disk
    img[28:31, 41] = 0
    labeled, n = ndimage.label(img)
    assert n == 2

    centers, _, relabeled = cs.separate_objects(img, 15, centers_only=False)
    for instance in range(1, n + 1):
        assert np.count_nonzero(centers[labeled == instance]) == 1
    assert len(np.unique(relabeled[labeled == 2])) == 1
    assert relabeled[29, 44] != relabeled[30, 25]


def test_labels_never_span_multiple_instances():
    img, _ = utils.read_image('sample_images/image1.tif')
    labeled, n = ndimage.label(img)
    centers, _, relabeled = cs.separate_objects(img, 15, centers_only=False)

    assert np.array_equal(relabeled > 0, img > 0)
    assert set(np.unique(relabeled)) == set(np.unique(centers))
    for instance in range(1, n + 1):
        assert np.count_nonzero(centers[labeled == instance]) >= 1
    for label in np.unique(relabeled)[1:]:
        assert len(np.unique(labeled[relabeled == label])) == 1


def test_centers_only_and_empty_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    centers, center_distances, relabeled = cs.separate_objects(img, 5, centers_only=True)
    assert relabeled is None
    assert not centers.any() and not center_distances.any()
    _, _, relabeled = cs.separate_objects(img, 5, centers_only=False)
    assert not relabeled.any()


def test_invalid_max_filter_size():
    with pytest.raises(ValueError):
        cs.separate_objects(np.ones((5, 5), dtype=np.uint8), 0, centers_only=True)


@pytest.mark.parametrize('end', [(0, 0), (9, 0), (0, 9), (9, 9), (3, 7), (7, 3), (5, 5)])
def test_gaps_on_connecting_line_reaches_endpoint(end):
    mask = np.ones((10, 10), dtype=np.bool_)
    assert cs.gaps_on_connecting_line(mask, 5, 5, *end) == 0
    mask[end] = False
    assert cs.gaps_on_connecting_line(mask, 5, 5, *end) == 1


def test_majority_filter_removes_lone_pixel():
    labels = np.full((7, 7), 1, dtype=np.int32)
    labels[3, 3] = 2
    mask = np.ones_like(labels, dtype=np.bool_)
    assert (cs.majority_filter_within_instance(labels, mask, 1) == 1).all()


def test_cli_skips_existing_outputs(tmp_path):
    cs.separate_images_in_dir('sample_images', '', '.tif', tmp_path, 15, True, False)
    out = tmp_path / 'centers_image1.tif'
    assert out.is_file()
    assert not (tmp_path / 'center_based_relabeled_image1.tif').exists()
    # Centers exist from the run above, but the relabeled image must still be produced
    cs.separate_images_in_dir('sample_images', '', '.tif', tmp_path, 15, False, False)
    assert (tmp_path / 'center_based_relabeled_image1.tif').is_file()
