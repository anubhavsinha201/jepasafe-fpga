"""JEPASafe: a 1D CNN + LSTM human-activity classifier for FPGA deployment.

  cnnlstm_keras.py  the model that exports (Keras; hls4ml cannot parse the
                    PyTorch version's transposes)
  export_hls.py     Keras -> hls4ml -> Vitis HLS project
  demopart.py       PyTorch baseline; also builds the .npy dataset cache
"""

__version__ = "0.1.0"
