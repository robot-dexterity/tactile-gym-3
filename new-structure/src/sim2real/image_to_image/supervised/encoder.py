class AbstractLabelEncoder:

    def __init__(self, device):
        pass

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.

        Returns: torch tensor that will be predicted by the NN
        """
        pass

    def decode_label(self, outputs):
        """
        Process NN predictions to raw pose data, always decodes to cpu.

        Returns: Dict of np arrays in suitable format for downstream task.
        """
        pass
