import os


def create_dir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # print('{} already exist in {}'.format(os.path.basename(dirname), os.path.dirname(dirname)))
        pass


class ColorModeError(Exception):
    """Error to handle color mode."""

    def __init__(self, info):
        self.info = info

        print('mode must be specified as ({}) for grayscale and ({}) for color!'
              .format(self.info % self.info, self.info // self.info))


class ChessBoardError(Exception):
    """Chessboard error handler."""

    def __init__(self):
        print('\nOpenCV: cv2.error: calibrateCameraRO failed to calibrate!\n'
              'Corner detection failed. Load chessboard images with good contrast!')