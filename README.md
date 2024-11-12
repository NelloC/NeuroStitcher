# NeuroStitch

**NeuroStitch** is a powerful toolkit designed for semi-automated neuron reconstruction. It refines and assembles 3D neuronal segmentations derived from biological microscopy images, providing a streamlined solution for reconstructing complex neuronal structures.

---

## Features

- **Semi-Automated Stitching**  
  Combines manual and automated tools for efficient neuron reconstruction.
  
- **Attractor Point Analysis**  
  Leverages advanced algorithms to detect and connect neuronal endings using attractor points and growth vectors.

- **Interactive 3D Visualization**  
  A user-friendly interface for visualizing and managing neuron stitching in 3D, including cluster views and attractor analyses.

- **Customizable Pipelines**  
  Adaptable to various microscopy imaging datasets, supporting different neuronal imaging formats.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/NeuroStitch.git
   cd NeuroStitch

2. Install dependencies: 
   Ensure you have Python installed. Then, run:
   ```bash
    pip install -r requirements.txt
    # Run the application:
    python main.py

# Usage
  1. Launching the Tool
      Run the application to start the interactive interface:
        ```bash
        python main.py

## Key Functionalities

### Manual Stitching Mode
Allows users to manually select and stitch neuron pieces, providing flexibility in reconstructing complex structures.

### Automated Stitch Suggestions
Based on calculated growth vectors and attractor points, the tool suggests potential connections between neuron pieces.

### Visualization Options
Toggle between different visualization modes, including:
- Cluster views
- Growth vectors
- Attractor points
- 3D Neurite structures

### Keyboard Shortcuts
Enhance your workflow with these shortcuts:
- **Next Candidate**: `→` (Arrow Right)
- **Previous Candidate**: `←` (Arrow Left)
- **Accept Stitch**: `Enter`
- **Reject Stitch**: `Backspace`
- **Toggle Visualization**: `V`

## Workflow

1. **Load Data**: Import neuronal data from supported formats such as `.json` or `.swc` using the interface.
2. **Analyze Attractor Points**: Use the attractor points module to identify potential neurite connections.
3. **Manual Review and Stitching**: Review the suggested connections in manual mode and accept or reject them using keyboard shortcuts.
4. **Export Results**: Save the stitched neuron reconstruction to a file for further analysis or visualization.

      ```plaintext
      NeuroStitch/
      ├── assets/                   # Static files for the interface
      ├── data/                     # Sample neuron datasets
      ├── NeuroStitcher/            # Main source directory
      │   ├── attractor_points.py   # Handles attractor points calculations
      │   ├── clustering.py         # DBSCAN and clustering utilities
      │   ├── connector.py          # Establishes connections between neuron segments
      │   ├── controller.py         # Controls the stitching process
      │   ├── debug_components.py   # Debug utilities for development
      │   ├── interactive_stitch_visualizer.py  # 3D visualization tool
      │   ├── loaders.py            # Loading different data formats (e.g., JSON, SWC)
      │   ├── main.py               # Entry point of the application
      │   ├── neuron_stitcher.py    # Main stitching logic
      │   ├── stitching.py          # Stitching manager with scoring
      │   ├── storage.py            # Data storage utilities for neuron data
      │   ├── utils.py              # General utilities for computation
      │   ├── vector_calculations.py # Growth vector and direction vector calculations
      │   └── view.py               # Interface views and visualization helpers
      ├── venv/                     # Virtual environment (optional)
      ├── README.md                 # Documentation


## Contributing

We welcome contributions to enhance **NeuroStitch**! Here’s how you can contribute:

1. **Fork the repository**.
2. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature
3. **Commit your changes**:
   ```bash
   git commit -m "Add your feature"
4. **Push to the branch**:
   ```bash
   git push origin feature/your-feature
5. **Open a Pull Request** and describe your changes.


##Acknowledgments
NeuroStitch was developed with support from the Laboratory of Anatomy, Histology, and Neuroscience at the Universidad Autónoma de Madrid, which provided invaluable experimental datasets for testing and refining the tool.

