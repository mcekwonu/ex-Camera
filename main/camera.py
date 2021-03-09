#!/usr/bin/env python3

import argparse
import pickle
import sys

import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper import create_dir, ChessBoardError


class Camera:
    """2D camera calibration to compute the intrinsic and extrinsic properties."""

    def __init__(self, source_dir, target_dir, outfilename='camera_params', square_size=20, chessboard_size=(9, 7),
                 show=None, verbose=None):
        """Initialize the chessboard dimension and parameters

		Parameters:
		    source_dir: source image directory
			target_dir: output save directory
			square_size: actual size (dimension in world unit) of one square pattern of chessboard. Default is 20 mm
			outfilename: output directory to save images and results.
			chessboard_size: number of squares of internal horizontal and vertical edges. Default=(9, 7)
			show: (bool) display and save calibrated image. Default is None. if show=True, the calibrated image is
			        displayed and saved in specified target directory, otherwise images are saved.
			verbose: (bool) to print summary of calibration results. Default is None.
		
		Returns:
		    None
		"""
        self.ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        self.target_dir = os.path.abspath(os.path.join(self.ROOT_DIR, target_dir))
        self.square_size = square_size  # mm
        self.chessboard_size = chessboard_size
        self.show = show
        self._source_dir = source_dir
        self.filename = outfilename
        self.camera_params = dict()

        # Prepare object points like (0, 0, 0), (1, 0, 0), (2, 0, 0), ... (6, 5, 0)
        self.objpt = np.zeros((np.prod(self.chessboard_size), 3), dtype=np.float32)
        self.objpt[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2) * self.square_size
        self.objpts = []  # 3D points in real world space
        self.imgpts = []  # 2D points in image plane

        self.errors = list()
        self.masks = list()

        # Termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #self.show = show
        self.verbose =  verbose

        self.rot_mtx = list()

        self.num = len(os.listdir(self._source_dir))

    def calibrate(self):
        """Returns Intrinsic and Extrinsic properties of camera.
		Intrinsic properties of camera: focal length (fx, fy) and center (Cx, Cy)
		
		Extrinsic camera properties: distortion coefficients, rotation vector and translation vector
		"""
        assert os.path.exists(self._source_dir), \
            '"{}" must exist and contain calibration images.'.format(self._source_dir)

        for _, fname in tqdm(enumerate(os.listdir(self._source_dir)), total=self.num, desc='Loading chessboard images'):
            img = cv2.imread(os.path.join(self._source_dir, fname), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.masks.append(mask)

        for _, img in tqdm(enumerate(self.masks), total=self.num, desc='Detecting chessboard images'):
            ret, corners = cv2.findChessboardCorners(img, self.chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object point, image points after refining them
            if ret:
                self.objpts.append(self.objpt)
                # refine corner location based on criteria
                self.corners_ = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpts.append(self.corners_)
            else:
                raise ChessBoardError()

        print(f'\nCalibrating {self.num} valid chessboards images found {self._source_dir} ...\n')

        self.dims = img.shape[::-1]
        self.h, self.w = self.dims
        self.img_orig = img.copy()

        # Draw and display the corners
        cv2.drawChessboardCorners(self.img_orig, self.chessboard_size, self.corners_, ret)

        # Calibrate camera
        ret, self.cam_mtx, self.dist_coeff, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpts, self.imgpts,
                                                                               self.dims, None, None)
        create_dir(self.target_dir)

        # Mean reprojection error
        mean_err = 0

        for i in range(len(self.objpts)):
            r_mtx_, _ = cv2.Rodrigues(self.rvecs[i], jacobian=0)
            imgpts_, _ = cv2.projectPoints(self.objpts[i], self.rvecs[i], self.tvecs[i], self.cam_mtx, self.dist_coeff)
            error = cv2.norm(self.imgpts[i], imgpts_, cv2.NORM_L2) / len(imgpts_)
            mean_err += error
            self.errors.append(error)
            self.rot_mtx.append(r_mtx_)
        self.mean_err = mean_err / len(self.objpts)

        if self.show:
            plt.imshow(self.img_orig, cmap='hot')
            plt.axis('off')
            plt.savefig(f'{self.target_dir}/{self.filename}_calibrated.png', dpi=1000)

            pos = np.arange(self.num)
            plt.figure()
            plt.bar(pos, self.errors, width=0.6, color='blue')
            plt.axhline(self.mean_err, color='red', linestyle='--')
            plt.ylabel('Mean error in Pixels')
            plt.xlabel('Image')
            plt.savefig(f'{self.target_dir}/{self.filename}_mean_error.png', dpi=1000)
            plt.show()
        else:
            plt.imshow(self.img_orig, cmap='hot')
            plt.axis('off')
            plt.savefig(f'{self.target_dir}/{self.filename}_calibrated.png', dpi=1000)

            pos = np.arange(self.num)
            plt.figure()
            plt.bar(pos, self.errors, width=0.6, color='blue')
            plt.axhline(self.mean_err, color='red', linestyle='--')
            plt.ylabel('Mean error in Pixels')
            plt.xlabel('Image')
            plt.savefig(f'{self.target_dir}/{self.filename}_mean_error.png', dpi=1000)

        self.save_calibration_results()

    def save_calibration_results(self):
        # save params for export
        self.camera_params['Number of Pattern'] = self.num
        self.camera_params['Image Size'] = self.dims
        self.camera_params['World Points'] = np.shape(self.objpts)
        self.camera_params['WorldUnits'] = 'mm'
        self.camera_params['Mean Reprojection Error'] = '{:.2f}'.format(self.mean_err)
        self.camera_params['Intrinsic Camera Matrix'] = self.cam_mtx
        self.camera_params['Focal lengths (f_x, f_y)'] = [self.cam_mtx[0, 0], self.cam_mtx[1, 1]]
        self.camera_params['Radial Distortion Coefficients'] = [self.dist_coeff[0, 0], self.dist_coeff[0, 1],
                                                                self.dist_coeff[0, -1]]
        self.camera_params['Tangential Distortion Coefficients'] = [self.dist_coeff[0, 2], self.dist_coeff[0, 3]]
        self.camera_params['Estimated Skew'] = self.cam_mtx[0, 1]
        self.camera_params['Principal Points (C_x, C_y)'] = '[{:.4f}, {:.4f}]'.format(self.cam_mtx[0, -1],
                                                                                      self.cam_mtx[1, -1])
        self.camera_params['Aspect Ratio (fy/fx)'] = '{:.2f}'.format(self.cam_mtx[1, 1] / self.cam_mtx[0, 0])
        self.camera_params['Rotation Matrix'] = np.mean(self.rot_mtx, axis=0)
        self.camera_params['Translation Vectors'] = np.mean(self.tvecs, axis=0)

        # save camera parameters in pickle format
        with open('{}/{}'.format(self.target_dir, self.filename) + '.pkl', 'wb') as f:
            pickle.dump(self.camera_params, f, pickle.HIGHEST_PROTOCOL)

        # save camera parameters in txt format
        with open('{}/{}'.format(self.target_dir, self.filename) + '.txt', 'w') as f:
            f.write(str(self.camera_params))

        if self.verbose:
            print('Calibration Results:\n====================')
            print('Number of Pattern: {}'.format(self.num))
            print('Image Size: {}'.format(self.dims))
            print('Mean Reprojection Error: {:.2f}'.format(self.mean_err))
            print('Intrinsic Camera Matrix:\n{}'.format(self.cam_mtx))
            print('Radial Distortion Coefficients: {}'.format([self.dist_coeff[0, 0], self.dist_coeff[0, 1],
                                                               self.dist_coeff[0, -1]]))
            print('Tangential Distortion Coefficients: {}'.format([self.dist_coeff[0, 2], self.dist_coeff[0, 3]]))
            print('Rotation Matrix:\n{}'.format(np.mean(self.rot_mtx, axis=0)))
            print('Translation Vectors:\n{}'.format(np.mean(self.tvecs, axis=0)))

    def compute_undistort(self):
        """Returns undistorted image.
		"""
        # Randomly choose an image from the calibration images and return undistorted image.
        n, fnames = self.get_sorted_images(self._source_dir)
        img_ = cv2.imread(os.path.join(self._source_dir, fnames[n]), cv2.IMREAD_GRAYSCALE)

        h, w = img_.shape[:2]  # get image height and width
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.dist_coeff, (w, h), 1, (w, h))

        dst = cv2.undistort(img_, self.cam_mtx, self.dist_coeff, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x + w//2:w]
        cv2.imwrite('{}/{}_undistorted_{}.png'.format(self.target_dir, self.filename, n), dst)

    @staticmethod
    def get_sorted_images(source_dir):
        """Returns index and sorted image list."""
        fnames = sorted(os.listdir(source_dir),
                        key=lambda x: int(os.path.splitext(x)[0]) if x.isdigit() else os.path.splitext(x)[0])
        index = np.random.randint(len(fnames))
        return index, fnames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard Camera Calibration')
    parser.add_argument('--source', type=str, required=True, help='Input image directory')
    parser.add_argument('--target', type=str, required=True, help='Output directory to save reports')
    parser.add_argument('--filename', type=str, required=True, help='Name to save the camera parameter')
    parser.add_argument('--sq_size', type=float, required=True, help='Chessboard square size')
    parser.add_argument('--cb_size', type=int, nargs="+", required=True, default=(9, 7),
                        help='Number of squares in chessboard, default=(9, 7)')
    parser.add_argument('--show', type=bool, default=None, required=False,
                        help='Display calibrated images, Reprojection error plot, undistorted image and save it.')
    parser.add_argument('--verbose', type=bool, default=None, required=False,
                        help='Print summary of calibration results')

    args = parser.parse_args()

    cam = Camera(args.source, args.target, args.filename, args.sq_size, tuple(args.cb_size), args.show, args.verbose)
    cam.calibrate()
    cam.compute_undistort()


