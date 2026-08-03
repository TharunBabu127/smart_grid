from utils import *
import os

signal = load_iq_signal(
    os.path.join(
        IQ_DATASET,
        "Healthy",
        "Healthy_0000.npy"
    )
)

plot_signal(signal, "Healthy IQ Signal")