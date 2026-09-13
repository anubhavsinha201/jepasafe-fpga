# JEPASafe: A Transformer JEPA-Based FPGA Accelerator for Real-Time Women's Safety

## A Complete Project Documentation & Research Guide

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Technical Architecture](#-technical-architecture)
3. [Hardware Stack](#-hardware-stack)
4. [Software Stack](#-software-stack)
5. [Implementation Timeline](#-implementation-timeline)
6. [Dataset Resources](#-dataset-resources)
7. [Advanced Features](#-advanced-features)
8. [Learning Resources](#-learning-resources)
9. [Research References](#-research-references)
10. [Budget Breakdown](#-budget-breakdown)
11. [Glossary](#-glossary)
12. [Quick Commands & Snippets](#-quick-commands--snippets)
13. [Project Build Log Template](#-project-build-log-template)

---

## 🎯 Project Overview

### What You're Building

A **proof-of-concept FPGA-based AI inference system** that demonstrates the complete workflow of taking a trained neural network (specifically a 1D CNN + JEPA architecture) and deploying it onto low-cost hardware for real-time anomaly detection in a women's safety pendant application.

### Core Concept

**Train → Quantize → hls4ml → HLS → FPGA Tools → Deploy on FPGA**

The system works in three stages:
1. **Sense**: An accelerometer (MPU6050) continuously measures movement.
2. **Think**: The FPGA runs a compact AI model (1D CNN + JEPA) to analyze motion data in real-time.
3. **Act**: If the AI detects an anomaly (fall, violent movement), it triggers an alert (LED/Bluetooth).

### Key Deliverables

| Component | Description |
|-----------|-------------|
| **Trained & Quantized Model** | Small neural network converted to integer precision |
| **FPGA Bitstream** | Synthesized hardware design for Tang Nano 9K |
| **Inference Engine** | FPGA running model and making predictions in real-time |
| **Test Harness** | Sensor connection or software testbench feeding data |
| **Alert Mechanism** | LED or Bluetooth alert on anomaly detection |

---

## 🏗️ Technical Architecture

### The Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR COMPUTER (PC/Laptop)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │    STEP 1        │    │    STEP 2        │    │    STEP 3        │      │
│  │  Train Model     │───▶│   Quantize       │───▶│  hls4ml         │      │
│  │  (PyTorch/Keras) │    │   (QKeras)       │    │  (Convert to C++)│      │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘      │
│         │                       │                        │                 │
│         ▼                       ▼                        ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              STEP 4: High-Level Synthesis (HLS)                 │      │
│  │         (Vivado HLS / Gowin HLS - Converts C++ to RTL/Verilog) │      │
│  └────────────────────────────────────────────────────────┬─────────┘      │
│                                                          │                 │
│  ┌───────────────────────────────────────────────────────▼──────────┐      │
│  │              STEP 5: FPGA Toolchain (Gowin IDE)                  │      │
│  │    - Synthesize Verilog to Logic Gates                           │      │
│  │    - Place & Route (Map to FPGA resources)                       │      │
│  │    - Generate Bitstream (.fs file)                               │      │
│  └────────────────────────────────────────────────────────┬─────────┘      │
│                                                          │                 │
└──────────────────────────────────────────────────────────┼─────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TANG NANO 9K FPGA BOARD                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │    STEP 6        │    │    STEP 7        │    │    STEP 8        │      │
│  │  Program FPGA    │───▶│   FPGA Runs      │───▶│   LED/Alarm      │      │
│  │  (Load Bitstream)│    │   AI Inference   │    │   Triggers!      │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                              ▲                                             │
│                              │                                             │
│                    ┌─────────┴─────────┐                                  │
│                    │  MPU6050 Sensor    │                                  │
│                    │  (Accelerometer)   │                                  │
│                    └───────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The JEPA + 1D CNN Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JEPASafe Model Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           INPUT: 50-100 samples from 6-axis IMU          │  │
│  │              (Accelerometer + Gyroscope)                 │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │          STAGE 1: 1D CNN Front-End (Feature Extractor)   │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Conv1D(filters=8, kernel_size=3, activation='relu')│  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │        MaxPooling1D(pool_size=2)                   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │          STAGE 2: Transformer JEPA (Predictor)           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    TransformerEncoderLayer                          │  │  │
│  │  │    - 2 Attention Heads                              │  │  │
│  │  │    - Model Dimension: 16                            │  │  │
│  │  │    - Feed Forward Dimension: 64                     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           OUTPUT: Anomaly Score (0 to 1)                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │       Dense(1, activation='sigmoid')               │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Threshold: If score > 0.8 → ALERT TRIGGERED!                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔌 Hardware Stack

### Required Components

| Component | Purpose | Approx. Cost | Link/Supplier |
|-----------|---------|--------------|---------------|
| **Sipeed Tang Nano 9K** | FPGA Development Board | ₹1,200 - ₹2,149 | [Probots](https://probots.co.in/sipeed-tang-nano-9k-fpga-development-board.html), [Rees52](https://rees52.com/products/sipeed-tang-nano-9k-fpga-board) |
| **MPU6050 (GY-521)** | 6-Axis Accelerometer + Gyroscope | ₹150 | Amazon, Robu, local electronics store |
| **HC-08 Bluetooth Module** | BLE Communication (Optional) | ₹250 | Robu, Amazon |
| **Jumper Wires & Breadboard** | Connections | ₹100 | Local electronics store |
| **USB-C Cable** | Power & Programming | Included with board | - |

**Total: ~₹1,700 - ₹2,000**

### Board Specifications (Tang Nano 9K)

| Specification | Detail |
|---------------|--------|
| **FPGA Chip** | Gowin GW1NR-9 |
| **Logic Units** | 8,640 LUT4 |
| **DSP Slices** | 20 |
| **Memory** | 64M-bit PSRAM, 32M-bit SPI Flash |
| **Onboard** | 6 programmable LEDs, USB-C programming, 27MHz clock |
| **Programming** | USB-C (onboard BL702 chip for USB-JTAG) |

### Connection Diagram

#### MPU6050 (I2C Interface)

| MPU6050 Pin | Tang Nano 9K Pin | Description |
|-------------|------------------|-------------|
| `VCC` | `3.3V` | Power supply (Do NOT use 5V) |
| `GND` | `GND` | Common ground |
| `SCL` | Pin 38 (or any GPIO) | I2C Clock line |
| `SDA` | Pin 27 (or any GPIO) | I2C Data line |

#### HC-08 (UART Interface)

| HC-08 Pin | Tang Nano 9K Pin | Description |
|-----------|------------------|-------------|
| `VCC` | `3.3V` | Power supply |
| `GND` | `GND` | Common ground |
| `TXD` | Pin 28 (or any GPIO) | Transmit data to FPGA |
| `RXD` | Pin 29 (or any GPIO) | Receive data from FPGA |

---

## 💻 Software Stack

### Development Tools

| Tool | Purpose | License | Link |
|------|---------|---------|------|
| **PyTorch / TensorFlow** | Model Training | Free (Open-source) | [PyTorch](https://pytorch.org), [TensorFlow](https://tensorflow.org) |
| **QKeras** | Quantization | Free (Open-source) | [GitHub](https://github.com/google/qkeras) |
| **hls4ml** | ML to HLS Conversion | Free (Open-source) | [GitHub](https://github.com/hls-fpga-machine-learning/hls4ml) |
| **Gowin EDA (IDE)** | FPGA Synthesis & Programming | Free (Education License) | [Gowin Semiconductor](https://www.gowinsemi.com) |
| **Yosys / NextPNR** | Open-source FPGA Toolchain | Free (Open-source) | [YosysHQ](https://github.com/YosysHQ) |
| **openFPGALoader** | FPGA Programming | Free (Open-source) | [GitHub](https://github.com/trabucayre/openFPGALoader) |
| **Icarus Verilog** | Simulation | Free (Open-source) | [GitHub](https://github.com/steveicarus/iverilog) |
| **GTKWave** | Waveform Viewer | Free (Open-source) | [GitHub](https://github.com/gtkwave/gtkwave) |

### Python Environment Setup

```bash
# Create a Python virtual environment
python -m venv fpga_project
source fpga_project/bin/activate  # On Windows: fpga_project\Scripts\activate

# Install required packages
pip install tensorflow keras qkeras hls4ml
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision  # Optional, if using PyTorch
```

---

## 📅 Implementation Timeline

### 2-Month Accelerated Plan

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| **Week 1** | Toolchain Setup & Hello World | LED Blinker project running on Tang Nano 9K |
| **Weeks 2-3** | Model Training & Quantization | Trained and quantized model in Python |
| **Weeks 4-6** | FPGA Deployment with hls4ml | Synthesized bitstream for Tang Nano 9K |
| **Weeks 7-8** | System Integration & Testing | Functional prototype with LED alert |

### 6-Month Research Plan

| Phase | Weeks | Goal |
|-------|-------|------|
| **Phase 1** | 1-4 | Foundation & Toolchain Setup |
| **Phase 2** | 5-10 | JEPA Model Design & Training |
| **Phase 3** | 11-16 | FPGA Deployment with hls4ml |
| **Phase 4** | 17-20 | Sensor Integration & Testing |

---

## 📊 Dataset Resources

### Available Datasets for Activity Recognition & Fall Detection

| Dataset | Description | Sensor | Activities | Link |
|---------|-------------|--------|------------|------|
| **BandX-Activity** | 33 volunteers, wristband with MPU6050, 20 Hz | MPU6050 | Walking, Jogging, Typing, 7 total | [Zenodo](https://zenodo.org) |
| **EdgeFall** | 1500 trials of falls and daily activities | LSM6DS3 (similar to MPU6050) | Falls (forward/backward/lateral), Daily activities | [IEEE DataPort](https://ieee-dataport.org) |
| **UCI HAR** | 30 participants, smartphone at waist, 50 Hz | Accelerometer + Gyroscope | 6 activities (walking, sitting, etc.) | [UCI Repository](https://archive.ics.uci.edu), [Kaggle](https://kaggle.com) |

### Pro Tips

1. **Start Simple**: Begin with UCI HAR for easy debugging and comparison with existing literature.
2. **Match Your Sensor**: BandX-Activity uses the exact MPU6050 sensor you'll deploy.
3. **Look for Code**: BandX-Activity includes code notebooks for 1D CNN training.

---

## 🚀 Advanced Features to Incorporate

### AI & Deep Learning Enhancements

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| **Knowledge Distillation** | Train tiny "student" model to mimic large "teacher" | High (Proven on similar hardware) |
| **Hardware-Aware NAS** | Algorithm auto-designs optimal network for your FPGA | Medium |
| **Bayesian Neural Networks** | Provides confidence/uncertainty in predictions | Medium |
| **Spiking Neural Networks** | Ultra-low power using sparse "spikes" | Medium-High |
| **Continual Learning / HDC** | Device adapts to specific user patterns over time | Medium |
| **Reinforcement Learning** | Dynamic power management, adaptive decision making | Medium |
| **Federated Learning** | Privacy-preserving model improvement | Medium-High |

### Multi-Modal Sensing

- **Audio Analysis**: Add microphone for scream detection
- **Heart Rate Monitoring**: ECG/PPG sensor for distress detection
- **Environmental Sensors**: Temperature, humidity, GPS
- **Discreet Camera**: Evidence capture (requires offloading)

### System Features

- **Energy Harvesting**: Solar or triboelectric nanogenerator (TENG)
- **Covert Activation**: Gesture or voice-activated alerts
- **Self-Defense Mechanisms**: Loud alarm, electric shocker
- **GSM/4G Communication**: Independent SOS alerts
- **Advanced Encryption**: Secure data transmission

---

## 📚 Learning Resources

### Verilog & FPGA

| Resource | Type | Platform |
|----------|------|----------|
| **Hardware Modeling Using Verilog** | Video Course | YouTube (IIT Kharagpur) |
| **Digital Design and Computer Architecture** | Book | Harris & Harris |
| **FPGA Programming for Beginners** | Book | Russell Merrick |
| **Coddy.tech** | Interactive Course | Website |
| **EDA Playground** | Online Simulator | Website |
| **Sipeed Tang Nano 9K Wiki** | Documentation | [Sipeed Wiki](https://wiki.sipeed.com) |
| **Lushay Labs Tang Nano Blog** | Tutorials | [Lushay Labs](https://learn.lushaylabs.com) |

### AI & Edge ML

| Resource | Type | Platform |
|----------|------|----------|
| **hls4ml GitHub** | Code Repository | [GitHub](https://github.com/hls-fpga-machine-learning/hls4ml) |
| **hls4ml Tutorials** | Step-by-Step Guides | [GitHub](https://github.com/hls-fpga-machine-learning/hls4ml-tutorial) |
| **Example Models** | Pre-built Examples | [GitHub](https://github.com/hls-fpga-machine-learning/example-models) |
| **QKeras Documentation** | Quantization Guide | [GitHub](https://github.com/google/qkeras) |

### Reinforcement Learning

| Resource | Type | Platform |
|----------|------|----------|
| **RLtools** | C++ RL for Edge Devices | [GitHub](https://github.com/RLtools) |
| **SECDA Methodology** | FPGA Co-Design | Research Paper |
| **FPGA-Gym** | Parallel RL on FPGA | Research Project |
| **Simple Q-Learning Tutorial** | Foundation | Multiple online sources |

---

## 📄 Research References

### JEPA & Self-Supervised Learning

1. **LeCun, Y. (2022).** "A Path Towards Autonomous Machine Intelligence." OpenReview.
2. **Assran, M., et al. (2023).** "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR.

### Time-Series JEPA

3. **TS-JEPA** - Time Series Joint Embedding Predictive Architecture.
4. **FinJEPA** - Financial Time Series JEPA.
5. **Brain-JEPA** - Biomedical fMRI Data JEPA.
6. **CHARM** - Industrial Sensor Anomaly Detection.

### FPGA AI Deployment

7. **StrikeWatch** - Transformer on iCE40UP5K FPGA.
8. **STORM** - 19.7k-parameter Transformer for Human Activity Recognition.
9. **CXL-SpecKV** - FPGA-based KV-Cache Management (Best Paper Nomination, FPGA '26).

### Women's Safety & Wearables

10. **Recent Review** - "Assault Detection and Technologies" (2026).
11. **FPGA-Based Hybrid GCN-Transformer** - 3D Pose Estimation (2026).

---

## 💰 Budget Breakdown

| Item | Cost (₹) | Source |
|------|----------|--------|
| Sipeed Tang Nano 9K | ₹1,200 - ₹2,149 | Probots, Rees52 |
| MPU6050 (GY-521) | ₹150 | Amazon, Robu |
| HC-08 Bluetooth | ₹250 | Robu, Amazon |
| Jumper Wires & Misc | ₹100 | Local store |
| **Total** | **₹1,700 - ₹2,650** | |

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **FPGA** | Field-Programmable Gate Array - reprogrammable hardware chip |
| **LUT** | Look-Up Table - basic logic unit in FPGA |
| **DSP** | Digital Signal Processor - specialized processing unit in FPGA |
| **HLS** | High-Level Synthesis - converting C++ to hardware design |
| **RTL** | Register Transfer Level - hardware description level |
| **Bitstream** | Binary configuration file that programs an FPGA |
| **Quantization** | Converting floating-point to integer precision |
| **JEPA** | Joint-Embedding Predictive Architecture - self-supervised learning framework |
| **1D CNN** | 1-Dimensional Convolutional Neural Network - for time-series data |
| **Autoencoder** | Neural network that learns to reconstruct its input |
| **Anomaly Detection** | Identifying patterns that do not conform to expected behavior |
| **Inference** | Running a trained model to make predictions |
| **I2C** | Inter-Integrated Circuit - 2-wire communication protocol |
| **UART** | Universal Asynchronous Receiver-Transmitter - serial communication |
| **BLE** | Bluetooth Low Energy - power-efficient wireless communication |

---

## 🔧 Quick Commands & Snippets

### Verilog LED Blinker for Tang Nano 9K

```verilog
module led (
    input sys_clk,      // 27MHz clock input
    input sys_rst_n,    // Reset (active low)
    output reg [5:0] led // 6 LEDs on the board
);

reg [23:0] counter;

// Counter: counts up to 0.5 seconds at 27MHz
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n)
        counter <= 24'd0;
    else if (counter < 24'd1349_9999)  // 0.5s delay
        counter <= counter + 1'b1;
    else
        counter <= 24'd0;
end

// LED shift register: rotates LEDs every 0.5 seconds
always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n)
        led <= 6'b111110;
    else if (counter == 24'd1349_9999)
        led[5:0] <= {led[4:0], led[5]};  // Rotate left
    else
        led <= led;
end

endmodule
```

### Python Autoencoder Training Skeleton

```python
import tensorflow as tf
from tensorflow import keras

# A tiny autoencoder for 6-input data (accelerometer: x,y,z for acceleration + gyro)
input_dim = 6
bottleneck_dim = 2

# Encoder
input_layer = keras.Input(shape=(input_dim,))
encoded = keras.layers.Dense(4, activation='relu')(input_layer)
bottleneck = keras.layers.Dense(bottleneck_dim, activation='relu')(encoded)

# Decoder
decoded = keras.layers.Dense(4, activation='relu')(bottleneck)
output_layer = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Step 2: Train on "Normal" Data Only
normal_data = [...]  # Your collected sensor data from walking/sitting
autoencoder.fit(normal_data, normal_data, epochs=50)

# Save the model for hls4ml
autoencoder.save('quantized_autoencoder.h5')
```

### hls4ml Model Conversion

```python
import hls4ml

# Load your trained model
model = keras.models.load_model('quantized_autoencoder.h5')

# Create HLS configuration
config = hls4ml.utils.config_from_keras_model(
    model, 
    granularity='name',
    default_precision='ap_fixed<8,4>'  # 8-bit quantization
)

# Convert to HLS project
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls_project',
    project_name='autoencoder'
)

# Compile the HLS project
hls_model.compile()
```

---

## 📝 Project Build Log Template

### Week 1: Toolchain Setup

**Date:** _______________

**Tasks Completed:**
- [ ] Ordered Tang Nano 9K
- [ ] Installed Gowin IDE
- [ ] Installed Python environment
- [ ] Ran LED Blinker tutorial

**Challenges Faced:**
1. 
2. 

**Solutions:**
1. 
2. 

---

### Week 2: Verilog Learning

**Date:** _______________

**Tasks Completed:**
- [ ] Completed Verilog basics course
- [ ] Understood always blocks, registers, combinational logic
- [ ] Simulated counter module

**Challenges Faced:**
1. 
2. 

**Solutions:**
1. 
2. 

---

### Week 3-4: Model Training

**Date:** _______________

**Tasks Completed:**
- [ ] Downloaded dataset (UCI HAR / EdgeFall / BandX)
- [ ] Trained autoencoder in Python
- [ ] Applied quantization with QKeras
- [ ] Tested anomaly detection threshold

**Challenges Faced:**
1. 
2. 

**Solutions:**
1. 
2. 

---

### Week 5-6: FPGA Deployment

**Date:** _______________

**Tasks Completed:**
- [ ] Generated HLS C++ code with hls4ml
- [ ] Synthesized in Gowin IDE
- [ ] Generated bitstream
- [ ] Debugged resource issues

**Challenges Faced:**
1. 
2. 

**Solutions:**
1. 
2. 

---

### Week 7-8: Integration & Testing

**Date:** _______________

**Tasks Completed:**
- [ ] Wrote I2C controller for MPU6050
- [ ] Integrated sensor with AI model
- [ ] Implemented LED alert
- [ ] Tested fall detection

**Challenges Faced:**
1. 
2. 

**Solutions:**
1. 
2. 

---

## 🔗 Quick Links

| Resource | URL |
|----------|-----|
| **Tang Nano 9K Wiki** | https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-9K/Nano-9K.html |
| **hls4ml** | https://github.com/hls-fpga-machine-learning/hls4ml |
| **hls4ml Tutorials** | https://github.com/hls-fpga-machine-learning/hls4ml-tutorial |
| **QKeras** | https://github.com/google/qkeras |
| **Gowin EDA** | https://www.gowinsemi.com |
| **Lushay Labs Tutorials** | https://learn.lushaylabs.com/tang-nano-series/ |
| **UCI HAR Dataset** | https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones |
| **EdgeFall Dataset** | https://ieee-dataport.org/open-access/edgefall |
| **BandX-Activity** | https://zenodo.org/record/8400931 |

---

## 📝 Final Checklist

Before presenting your project, ensure you have:

- [ ] Working LED Blinker on Tang Nano 9K
- [ ] Trained and quantized model in Python
- [ ] Synthesized bitstream for the model
- [ ] Functional inference on the FPGA
- [ ] Sensor integration (or testbench simulation)
- [ ] Working alert mechanism (LED/Bluetooth)
- [ ] Documentation and presentation materials
- [ ] All code on GitHub (public or private)
- [ ] A demo video (2-3 minutes)
- [ ] A clear explanation of the flow: Train → Quantize → hls4ml → FPGA

---

**End of Document**

*Created: August 2026*
*For: NIT Trichy Electron Devices Project*
*Project: JEPASafe - Transformer JEPA on FPGA for Women's Safety*