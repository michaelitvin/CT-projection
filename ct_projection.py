import copy
import os
import pydicom
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from bresenham import bresenhamline


def orientation_to_affine_matrix(position, direction_cosines, voxel_spacing):
    """
    Convert the DICOM position/orientation representation into an affine transformation matrix.
    Based on Equation C.7.6.2.1-1 from the DICOM standard, with the addition of also handling the z-axis by scaling.
    https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037
    :param list[float] position: 3 numbers, see the link above.
    :param list[float] direction_cosines: 6 numbers, see the link above.
    :param list[float] voxel_spacing: 3 numbers, see the link above. (2 first are pixel spacing,
                                                                      the last is slice thickness)
    :return np.ndarray: Corresponding affine transformation matrix
    """
    direction_cosines = np.array(direction_cosines).flatten()
    position = np.array(position)
    m = np.zeros([4, 4])
    m[:-1, 0] = voxel_spacing[0] * direction_cosines[:3]
    m[:-1, 1] = voxel_spacing[1] * direction_cosines[3:]
    m[2, 2] = voxel_spacing[2]
    m[:-1, 3] = position
    m[3, 3] = 1
    return m


def test_orientation_to_affine_matrix():
    from numpy.testing import assert_array_equal
    m = orientation_to_affine_matrix([.1, .2, .3], [1, 2, 3, 4, 5, 6], [2, 3, 4])
    assert_array_equal(m, np.array([[2., 12., 0., 0.1],
                                    [4., 15., 0., 0.2],
                                    [6., 18., 4., 0.3],
                                    [0., 0., 0., 1.]]))


def load_dicom_ct_scan_from_folder(folder):
    """
    Inspired by https://github.com/pydicom/pydicom/blob/master/examples/image_processing/reslice.py
    :param str folder: see proj_ct_to_xray.
    :return tuple(np.ndarray, list(float), list(float), list(float)): ct_img3d, voxel_spacing, position, orientation
    """
    # load the DICOM files
    filenames = [fn for fn in os.listdir(folder) if fn[-4:] == '.dcm']
    files = []
    print('Loading {} files from {}'.format(len(filenames), folder))
    for fname in filenames:
        files.append(pydicom.read_file(os.path.join(folder, fname)))

    print("Done loading {} files".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # voxel aspects and slice position/orientation, assuming all slices are the same (except for the position z value)
    def to_float(arr):
        return list(map(float, arr))

    voxel_spacing = to_float(list(slices[0].PixelSpacing) + [str(slices[0].SliceThickness)])
    position = to_float(slices[0].ImagePositionPatient)
    orientation = to_float(slices[0].ImageOrientationPatient)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    print("Shape: {}".format(img_shape))
    ct_img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        ct_img3d[:, :, i] = img2d

    # Treat negative values as zeros (looks like this is the intention in the given data)
    ct_img3d[ct_img3d < 0] = 0.

    return ct_img3d, voxel_spacing, position, orientation


class SensorBoard:
    def __init__(self, transform, resolution):
        """
        Creates a SensorBoard object with the given positioning parameters.
        An identity transform means that the board is on the xz plane with its left-top pixel at the origin,
        rows growing in the z direction and columns growing in the x direction with unit scale.
        :param np.ndarray transform: Affine 3D transformation defining the translation of the left-top pixel
                                     center in mm from the origin, rotation of the board in 3D space, and scaling
                                     according to the pixel spacing.
                                     The expected shape is (4,4).
        # :param list[float] pixel_spacing: Physical distance between the center of each pixel, specified by a numeric
        #                                   pair - adjacent row spacing (delimiter) adjacent column spacing in mm.
        :param tuple[int] resolution: Shape of the image produced by the board (rows, columns)
        """
        self.transform = transform
        self.resolution = resolution

    def pixel_to_3d_hg(self, px):
        """
        Transform the given pixel coordinates into the corresponding 3D position.
        :param list[float] px: Coordinates of a pixel on the sensor board
        :return np.ndarray: Coordinates of the pixel in 3D (homogenous)
        """
        return np.matmul(self.transform, np.array([px[0], 0, px[1], 1]))


def to_hg(v, is_point=True):
    """
    Converts a point or vector to homogenous coordinates.
    """
    return np.concatenate((v, [float(is_point)]))


def proj_ct_to_xray_from_folder(light_pos, board, ct_scan_folder):
    """
    Creates an image of the projection of the given CT volume onto the board plane, w.r.t. the X-ray light position.
    :param list[float] light_pos: Position of the X-ray light source
    :param SensorBoard board: Parameters of the sensor board
    :param str ct_scan_folder: Path to a folder containing *.dcm files of the CT scan.
    :return np.ndarray: Projected 2D X-ray image
    """
    # Load CT scan and its params
    ct_img3d, voxel_spacing, ct_position, ct_orientation = load_dicom_ct_scan_from_folder(ct_scan_folder)

    # Calculate the matrix of the CT volume and its inverse. The inverse should map the CT volume to have a corner
    # at the origin, be aligned with the axes, and have voxels of unit size (1x1x1).
    ct_matrix = orientation_to_affine_matrix(ct_position, ct_orientation, voxel_spacing)

    return proj_ct_to_xray(light_pos, board, ct_matrix, ct_img3d)


def proj_ct_to_xray(light_pos, board, ct_matrix, ct_img3d):
    """
    See proj_ct_to_xray_from_folder.
    """
    ct_matrix_inv = np.linalg.inv(ct_matrix)
    # Transform the X-ray light and board to a the coord system described above.
    light_pos_m = np.matmul(ct_matrix_inv, to_hg(light_pos))
    board_m = copy.copy(board)
    board_m.transform = np.matmul(ct_matrix_inv, board.transform)
    # Calculate the X-ray image
    xray_img = np.zeros(board.resolution)
    for i, j in tqdm(list(np.ndindex(board.resolution))):
        _process_ray(i, j, xray_img, ct_img3d, board_m, light_pos, light_pos_m)
    return xray_img


def _process_ray(i, j, xray_img, ct_img3d, board_m, light_pos, light_pos_m):
    # Calculate the voxels along the line segment connecting the light source with the sensor board pixel
    px3d_m = board_m.pixel_to_3d_hg([i, j])
    # TODO efficiency: intersect with the CT volume before running Bresenham to avoid making redundant calculations
    voxel_list = bresenhamline(light_pos_m[np.newaxis, :3], px3d_m[:3], max_iter=-1)
    # (assuming the entire CT volume is between the light source and the board)
    valid_voxels = np.logical_and(voxel_list >= 0, voxel_list < ct_img3d.shape).all(axis=1)
    voxel_list = voxel_list[valid_voxels].astype(np.int)
    # Calculate the distance the ray travels per voxel
    px3d = board_m.pixel_to_3d_hg([i, j])
    ray_direction_m = px3d_m[:3] - light_pos_m[:3]
    ray_direction = px3d[:3] - light_pos
    principal_dimension = np.argmax(np.abs(ray_direction_m))
    # The distance is 1/cos(angle between the line segment and the principal direction)
    distance_per_voxel = np.abs(norm(ray_direction) /
                                ray_direction[principal_dimension])
    # Sum the voxels along the line segment
    voxel_values = ct_img3d[tuple(voxel_list.T)]  # index into the CT volume
    xray_img[i, j] = np.sum(voxel_values) * distance_per_voxel


def demo():
    import matplotlib.pyplot as plt
    ct_scan_folder = './Case2/'
    light_pos = [0, 1000, -140.]
    ds = 1  # downsample to run faster
    board_transform = np.array([[ds, 0, 0, -300],
                                [0, ds, 0, 0],
                                [0, 0, ds, -280],
                                [0, 0, 0, 1]])
    board = SensorBoard(transform=board_transform,
                        resolution=(600 // ds, 280 // ds))
    xray_im = proj_ct_to_xray_from_folder(light_pos, board, ct_scan_folder)
    plt.imshow(xray_im)
    plt.show()


if __name__ == '__main__':
    demo()
