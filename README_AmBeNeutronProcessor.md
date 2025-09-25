# AmBeNeutronEff.py - User Manual

## Overview

`AmBeNeutronEff.py` is a comprehensive, integrated analysis tool for AmBe (Americium-Beryllium) neutron detection experiments. This single Python script handles complete neutron analysis workflows including waveform processing, event selection, efficiency calculations, and data export.

## Quick Start

### Basic Usage (Interactive Mode)
```bash
python3 AmBeNeutronEff.py
```

The script will prompt you for:
- **Campaign**: 1 or 2
- **Run type**: AmBe/Outside_source/C1/C2/ClusterCuts  
- **Directory paths**: Data and waveform directories
- **Analysis options**: Save samples, plot distributions

### Example Session
```
What campaign is this? (1/2): 2
What type of run is this? (AmBe/Outside_source/C1/C2/ClusterCuts): AmBe
Use default directories? (y/n) [y]: y
Save waveform samples? (y/n) [n]: n  
Plot IC distributions? (y/n) [y]: y
Start analysis? (y/n) [y]: y
```

## Features

### **Complete Analysis Pipeline**
- **Waveform Analysis**: Processes raw PMT waveforms from ROOT files
- **Event Processing**: Applies physics cuts and identifies neutron candidates
- **Efficiency Calculation**: Computes detection efficiency across source positions
- **Data Export**: Generates CSV files for further analysis

### **Physics Cuts & Selection**
- **Cosmic Muon Rejection**: Time and energy-based cuts
- **Single Neutron Selection**: PE, charge balance, and cluster time cuts
- **Multiple Neutron Detection**: Identifies events with multiple neutron candidates
- **Prompt Event Identification**: Separates prompt from delayed events

### 📊 **Visualization & Output**
- **IC Distributions**: Integrated charge histograms (linear and log scale)
- **Waveform Samples**: Optional sample waveform plots for quality control
- **CSV Export**: Structured data output for external analysis tools
- **Summary Reports**: Comprehensive analysis summaries

## Configuration

### Default Parameters

The script uses two main configuration classes:

#### WaveformConfig (Waveform Analysis Parameters)
```python
pulse_start: 300 ns          # Integration window start
pulse_end: 1200 ns           # Integration window end
pulse_gamma: 400             # Minimum IC acceptance threshold
NS_PER_ADC_SAMPLE: 2         # Time resolution
ADC_IMPEDANCE: 50 Ω         # System impedance
```

#### CutCriteria (Event Selection Cuts)
```python
pe_max: 80                   # Maximum photoelectrons
ccb_max: 0.40               # Maximum charge balance
ct_min: 2000 ns             # Minimum cluster time
cosmic_ct_threshold: 2000 ns # Cosmic event time threshold
cosmic_pe_threshold: 80      # Cosmic event PE threshold
```

### Source Position Mapping
The script automatically maps run numbers to source positions:
- **Campaign 1**: Runs 4453-4670 with various (x,y,z) positions
- **Campaign 2**: Runs 5682-5791 with updated position mapping

## Output Files

### Automatic Directory Creation
The script creates these directories if they don't exist:
- `TriggerSummary/` - Summary CSV files
- `EventAmBeNeutronCandidatesData/` - Per-run analysis results
- `verbose/` - Plots and detailed outputs

### Generated Files

#### Summary Files
1. **`TriggerSummary/AmBeWaveformResults_{runinfo}.csv`**
   - Waveform analysis summary by source position
   - Columns: x_pos, y_pos, z_pos, accepted_events, rejected_events, total_waveforms

2. **`TriggerSummary/AmBeTriggerSummary_{runinfo}.csv`**
   - Efficiency summary for each source position
   - Columns: x_pos, y_pos, z_pos, total_events, cosmic_events, ambe_triggers, single_neutron_candidates, multiple_neutron_candidates

#### Per-Run Files (one pair per run)
3. **`EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_{run}.csv`**
   - Individual neutron candidates with full event information
   - Columns: clusterTime, clusterPE, clusterChargeBalance, clusterHits, hitT, hitQ, hitPE, hitID, sourceX, sourceY, sourceZ, eventID, eventTankTime

4. **`EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_{run}.csv`**
   - Prompt neutron events (cosmic-like)
   - Columns: prompt_clusterTime, prompt_clusterPE, prompt_clusterChargeBalance

#### Visualization Files (optional)
5. **`verbose/AllRuns_IC_adjusted_{runinfo}.pdf`** - IC distributions for all runs
6. **`verbose/IC_adjusted_AllEvents_{runinfo}.png`** - Combined IC histogram  
7. **`verbose/Log_IC_adjusted_AllEvents_{runinfo}.png`** - Log-scale IC histogram
8. **`verbose/IC_adjusted_AcceptedEvents_{runinfo}.png`** - Accepted events only

## File Pattern Recognition

### Campaign 1 Files
- **Pattern**: `AmBe_<run_number>_v<version>.ntuple.root`
- **Example**: `AmBe_4506_v3.ntuple.root`

### Campaign 2 Files  
- **Pattern**: `BeamCluster_<run_number>.root`
- **Example**: `BeamCluster_4708.root`

## Advanced Usage

### Programmatic Usage
```python
from AmBeNeutronEff import AmBeNeutronEfficiencyAnalyzer, WaveformConfig, CutCriteria
import re

# Create analyzer with custom configuration
config = WaveformConfig(pulse_start=250, pulse_end=1400)
cuts = CutCriteria(pe_max=60, ccb_max=0.35)
analyzer = AmBeNeutronEfficiencyAnalyzer(config, cuts)

# Run analysis
results = analyzer.run_complete_analysis_pipeline(
    data_directory='../AmBe_BeamCluster/',
    waveform_dir='../AmBe_waveforms/',
    file_pattern=re.compile(r'BeamCluster_(\d+)\.root'),
    campaign=2,
    runinfo='custom_analysis'
)
```

### Modular Analysis
```python
# Use individual components
analyzer = AmBeNeutronEfficiencyAnalyzer()

# Just waveform analysis
waveform_results, runs, files = analyzer.analyze_multiple_runs(
    data_directory='../data/',
    waveform_dir='../waveforms/',
    file_pattern=pattern,
    campaign=2
)

# Just event processing  
event_data = analyzer.load_event_data('file.root', which_tree=1)
processed_data, stats = analyzer.process_events_efficient(
    event_data, good_events, x_pos, y_pos, z_pos
)
```

## Troubleshooting

### Common Issues

1. **"No files found matching pattern"**
   - Check data directory path
   - Verify file pattern matches your files
   - Ensure files have correct naming convention

2. **Memory errors with large datasets**
   - The script includes automatic memory management
   - Consider processing smaller subsets if issues persist

3. **Missing ROOT files**
   - Verify both data and waveform directories exist
   - Check that run numbers match between directories

4. **Empty output files**
   - Check that waveform analysis found good events
   - Verify cut criteria aren't too restrictive

### Performance Tips

- **Large datasets**: The script automatically handles memory management
- **Visualization**: Disable IC plotting (`plot_ic_distributions=False`) for faster processing
- **Sample waveforms**: Only save samples for quality control, not production runs

## Technical Details

### Analysis Workflow
1. **File Discovery**: Scans data directory for matching files
2. **Waveform Analysis**: Processes PMT waveforms, applies quality cuts
3. **Event Loading**: Loads cluster and hit data from ROOT files
4. **Event Processing**: Applies physics cuts, identifies neutron candidates
5. **Data Export**: Saves results to CSV files
6. **Visualization**: Generates plots (if enabled)

### Physics Cuts Applied
- **Pulse Quality**: IC threshold, pulse shape requirements
- **Time Selection**: Cluster time windows for neutron/cosmic separation  
- **Energy Cuts**: Photoelectron thresholds
- **Topology**: Charge balance requirements
- **Multiplicity**: Single vs. multiple neutron identification

### Data Quality Features
- **Progress Monitoring**: Real-time progress bars
- **Error Handling**: Graceful handling of corrupted files
- **Memory Management**: Automatic cleanup and garbage collection
- **Validation**: Input parameter checking and validation

## Support

For questions or issues:
1. Check the terminal output for detailed error messages
2. Verify input file paths and formats
3. Review the generated CSV files for data validation
4. Check the `verbose/` directory for diagnostic plots
