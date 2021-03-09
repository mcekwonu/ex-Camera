"""Script to run camera calibration"""
from camera import Camera


camera = Camera(source_dir='/home/mce/Documents/bubble3D/calibration/Cam01', outfilename='Cam01',
                target_dir='results/camera', verbose=True)
camera.calibrate()
camera.compute_undistort()
