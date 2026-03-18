# MedTech Sprintathon - Setup Instructions

## Virtual Environment Setup

### Initial Setup
```bash
# Create virtual environment
python3 -m venv venv-sprint

# Activate virtual environment
source venv-sprint/bin/activate  # macOS/Linux
# or
.\venv-sprint\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Running Scripts
With the virtual environment activated:

```bash
# Run FFT analysis
python fft_analysis.py

# Run Butterworth filter test
python 5thButter.py
```

### Project Structure
- `fft_analysis.py` - FFT spectrum analysis and frequency tracking
- `5thButter.py` - Butterworth filter implementation and crepitus detection
- `requirements.txt` - Python package dependencies
- `example_data.csv` - Hardware signal data (timestamp, signal format)

### Key Parameters
- **Sampling Frequency**: 10,000 Hz
- **Crepitus Band**: 3000-3050 Hz (50 Hz bandwidth)
- **Filter Order**: 5th Order Butterworth

### Data Format
Input CSV from hardware:
```
Timestamp,Signal
0.00,2000
0.20,1993
...
```
- Timestamp: milliseconds from start
- Signal: voltage measurement (volts)
