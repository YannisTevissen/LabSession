# import all the required libraries

from diart.blocks import PipelineConfig
from diart import OnlineSpeakerDiarization
from diart.sources import FileAudioSource
from diart.inference import RealTimeInference
from diart import models
from diart.sinks import RTTMWriter
import torch
import time


def diart_diarization(file: str, device: str):
    start = time.time()
    token = "TOKEN_TO_REPLACE"
    seg_model = models.SegmentationModel.from_pyannote("pyannote/segmentation", token)
    emb_model = models.EmbeddingModel.from_pyannote("pyannote/embedding", token)

    config = PipelineConfig(
        # Set the model used in the paper
        segmentation=seg_model,
        embedding=emb_model,
        step=0.5,
        latency=5,
        tau_active=0.576,
        rho_update=0.422,
        delta_new=0.648,
        device=torch.device(device)
    )
    pipeline = OnlineSpeakerDiarization(config)
    source = FileAudioSource(file=file, sample_rate=16000)
    inference = RealTimeInference(pipeline, source, do_plot=False)
    inference.attach_observers(RTTMWriter(source.uri, "results.rttm"))
    inference()
    end = time.time()
    # print(f"diart using {device} diarization took {end - start} seconds to finish")
    return end - start


if __name__ == '__main__':
    path_to_file = "PATH_TO_FILE"
    diart_diarization(path_to_file, 'cpu')