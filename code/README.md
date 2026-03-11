\# Comparative Analysis of SMART, SMARTS and TVK



\[!\[Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

\[!\[NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)

\[!\[SciPy](https://img.shields.io/badge/SciPy-1.7+-orange.svg)](https://scipy.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\##  Overview

This project implements and compares three Multi-Criteria Decision Making (MCDM) methods:

\- \*\*SMART\*\* (Simple Multi-Attribute Rating Technique)

\- \*\*SMARTS\*\* (SMART with Swing weighting)

\- \*\*TVK\*\* (Theory of Criterion Importance)



The study applies these methods to two case studies and performs Monte Carlo sensitivity analysis.



\## Case Studies



\### 1. Amsterdam Hostel Selection

\- \*\*Source\*\*: Podinovskii (2022), p. 170-172

\- \*\*6 alternatives\*\*: Hostel Slotania, Aivengo, Shelter City, Flying Pig, Stayokay, Vita Nova

\- \*\*5 criteria\*\*: Beds (min), Location (max), Cleanliness (max), Comfort (max), Facilities (max)



\### 2. Preflib Dots Dataset (#00024)

\- \*\*Source\*\*: Preflib Mechanical Turk

\- \*\*4 alternatives\*\*: A(200 dots), B(209), C(218), D(227)

\- \*\*5 criteria\*\*: Borda, Approval Rate, Simpson, Copeland, Dodgson

\- \*\*Ground truth\*\*: A < B < C < D



\### 3. Investment Projects (Monte Carlo)

\- \*\*4 alternatives\*\*: Café, E-shop, App, Tutoring

\- \*\*4 criteria\*\*: Profit (benefit), Risk (cost), Cost (cost), Time (benefit)



\##  Installation



\### Prerequisites

\- Python 3.13 or higher

\- pip (Python package installer)



\### Setup

```bash

\# Clone the repository

git clone https://github.com/Majd-nakhla/report-Comparative-Analysis-of-SMART-SMARTS-and-TVK.git

cd report-Comparative-Analysis-of-SMART-SMARTS-and-TVK



\# Create virtual environment (optional but recommended)

python -m venv .venv

\# On Windows:

.venv\\Scripts\\activate

\# On Linux/Mac:

source .venv/bin/activate



\# Install dependencies

pip install -r requirements.txt

