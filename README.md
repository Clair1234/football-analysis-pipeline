# football-analysis-pipeline
Pipeline set up to gain interpretable insights of the tactics set up from football matches by analysing positions of players and the ball and a transformer's embedding


## Description
### Abstract
This document serves as a setup guide for a pipeline designed to analyse football tactical patterns using transformer-based machine learning models. The pipeline processes player positional data to extract meaningful tactical insights through embedding analysis and clustering techniques. The system combines spatial analysis, temporal dynamics, and deep learning approaches to identify distinct tactical configurations and team behaviours in football matches.

### Keywords
Football analytics, Transformer models, Tactical analysis, Player positioning, Clustering analysis, Sports data science, Spatial-temporal analysis, Latent representations, Team behaviour analysis.

### Subject areas
Sports analytics, Machine learning, Data science, Behavioural analysis,  Tactical intelligence.

## Required resources
### Hardware requirements
**Memory:** Minimum 16GB RAM recommended for large dataset processing 

**GPU:**  CUDA-compatible GPU recommended for transformer model training 

**Storage:** Sufficient space for Parquet/CSV files and model checkpoints 

**Processing:** Multi-core CPU for data preprocessing and clustering analysis 


### Software dependencies
	- Python 3.8+
	- Core libraries:
		- Polars (for efficient data loading and processing)
		- NumPy
		- Pandas
		- PyTorch (for transformer models)
		- Scikit-learn (for clustering and PCA)
		- Gfootball (for generating football match data)
	
	- Visualisation
		- Matplotlib
		- Seaborn

## Demonstration workflow
### Overall workflow
	- Data Preparation:
	
	Load football match data in Parquet or CSV format
	
	Implement batch loading system for memory-efficient processing
	
	Extract player positional features and ball tracking data
	
	
	- Model Training:
	
	Configure transformer architecture (embedding dim: 128, heads: 4, layers: 3)
	
	Enable mixed precision training (float16) for efficiency
	
	Implement incremental learning with file batching
	
	Monitor loss and RMSE convergence
	
	
	- Feature Extraction:
	
	Generate embeddings from trained transformer model
	
	Calculate spatial features (density, compactness, spread)
	
	Compute movement metrics (distance covered, velocity patterns)
	
	Extract formation characteristics
	
	- Clustering Analysis:
	
	Apply PCA for dimensionality reduction
	
	Perform clustering on different feature combinations:
	Positional clustering,
	Per-agent clustering,
	Combined embeddings,
	Hybrid spatial-embedding approach
	
	Generate either field-level or team-level analysis
	
	Evaluate dimension impact

### Run
Those are example of application of code.

### Collect data - run_data_collection.py
```
python run_data_collection.py

```
or there are parameters to collect specific data, policy helps pick a particular policy (all, random, aggresive, defensive, passer, possession, counter), random will always be included, it is the basic GRF action manager. Default is 'random'. The frames parameter is responsible for the number of frames in a match, a frame is 0.1s, default is 5000. The round parameter counts the number of rounds of simulation per policy, default is 100. The parameter output_dir is responsible for where the data is stored once simulated, default is './data/raw'.
```
python run_data_collection.py --policy 'all' --frames 5000 --round 5 --output_dir './data/raw'

```

### Enhance data - run_data_enhancement.py
In main function, change the following line to where the data is stored. The output will be in './data/processed'. 
```
path = './data/raw/' 
```
the run:
```
python run_data_enhancement.py

```

### Run EDA - run_EDA.py
In main function, change the following line to where the data is stored. The output will be in './analysis/EDA/'.
```
path = './data/raw/' 
```
the run:
```
python run_EDA.py

```

### Split into training and testing - run_data_split.py
You can change those parameters in the config at the top:
```
# --- Config ---
source_folder = "./data/processed"
train_folder = "./data/processed/training"
test_folder = "./data/processed/testing"
train_ratio = 0.8  # 80% for training, 20% for testing
seed = 42  # for reproducibility
```
then run:
```
python run_data_split.py

```

### Train a model - run_train_model.py
The parameter `FROM_SCRATCH` is in charge of either creating a new architecture (if True) or loading a model (else). Pick architecture dimension and adapt data folder on line 76:
```
# Collect files available
files = glob.glob('./training/*.parquet')
```
then run:
```
python run_train_model.py
```

### Test a model - run_test_model.py
In main, adapt `model_path` and `data_path` then run:
```
python run_test_model.py

```

### Run analysis - run_tactical_analysis.py
In main, adapt `model_path` and `data_path` then run:
```
python run_tactical_analysis.py

```

### Accessing outputs
There are two types of outputs that are located in the analysis folder:

**Visualisations:**
Training curves (loss and RMSE plots),
player movement heat maps,
team centroid trajectories,
clustering visualisations (PCA, t-SNE), 
attention maps for model interpretability

**Analysis Results:**
Cluster identification and labelling,
feature importance rankings,
z-score analysis for cluster discrimination,
silhouette scores for clustering quality

## Code structure
### Main directory structure


### Data files
**Input Formats:** CSV and Parquet files containing player positional data.

**Key Data Element for CSV files:**
Player coordinates (X, Y positions),
timestamps for temporal analysis,
team identifiers,
ball position tracking,
match metadata,


**Key Data Element for Parquet files:**
Spatial metrics (density, compactness, spread),
movement patterns (distance covered, velocity),
formation characteristics (hull area, balance metrics),
temporal dynamics (autocorrelation, persistence)

### Machine learning model architecture
#### Model components
	- Transformer core
		- Embedding dimension: 128
		- Attention heads: 4
		- Number of layers: 3
		- Mixed precision training (float16)
		- Automatic Mixed Precision (AMP) support
	
	- Input processing
		- Player position encoding
		- Temporal sequence handling
		- Multi-agent representation (22 players + ball)
	
	- Training strategy
		- Incremental batch learning
		- Memory-efficient data loading
		- Loss and RMSE monitoring
		- Validation split for over-fitting prevention
		- Early-stopping with a patience of 10 (after 10 epochs)

#### Interpretability components
	- Attention analysis
		- Player-to-player attention maps
		- Average attention patterns per cluster
		- Sample-level attention variability
	
	- Embedding analysis
		- Dimensionality reduction (PCA)
		- Feature importance scoring
		- Cluster-specific dimension analysis
		- Z-score discriminative analysis
	
	- Clustering methods
		- Positional clustering (39 clusters)
		- Per-agent clustering (23 clusters)
		- Combined embeddings (4 clusters for field-based and 8 for team-based )
		- Hybrid spatial-embedding approach (4 clusters for field-based and 8 for team-based )
		- Team-based vs field-based analysis
	
	- Validation metrics
		- Silhouette scores for cluster quality
		- Feature importance rankings
		- Cross-cluster pattern analysis
		- Tactical pattern interpretability
