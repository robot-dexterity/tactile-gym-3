import cv2

from data_collection.process_data.image_transforms import process_image


class SimSensor:
    def __init__(self, sensor_params={}, embodiment={}):
        self.sensor_params = sensor_params
        self.embodiment = embodiment

    def read(self):
        img = self.embodiment.get_tactile_observation()
        return img

    def process(self, outfile=None):
        img = self.read()
        img = process_image(img, **self.sensor_params)
        if outfile:
            cv2.imwrite(outfile, img)
        return img


class RealSensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params
        source = sensor_params.get('source', 0)
        exposure = sensor_params.get('exposure', -7)

        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):
            self.cam.read()  # Hack - initial camera transient

    def read(self):
        # self.cam.read()  # Hack - throw one away - buffering issue (note - halves frame rate!)
        _, img = self.cam.read()
        return img

    def process(self, outfile=None):
        img = self.read()
        img = process_image(img, **self.sensor_params)
        if outfile:
            cv2.imwrite(outfile, img)
        return img


class ReplaySensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params

    def read(self, outfile):
        img = cv2.imread(outfile)
        return img

    def process(self, outfile):
        img = self.read(outfile)
        img = process_image(img, **self.sensor_params)
        return img
