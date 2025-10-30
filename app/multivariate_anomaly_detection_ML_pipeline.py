import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class CycleBasedAnomalyDetection:
    """
    Anomaly Detection with proper cycle-based train/test split and interactive visualizations
    """
    
    def __init__(self, data_dir='anomaly_detection_data', test_size=0.2):
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.test_data_by_cycle = []  # Store test data organized by cycle
        
        # Load manifest
        with open(self.data_dir / 'dataset_manifest.json', 'r') as f:
            self.manifest = json.load(f)
        
        print("="*80)
        print("CYCLE-BASED ANOMALY DETECTION PIPELINE")
        print("="*80)
        print(f"Total cycles available: {self.manifest['total_cycles']}")
        print(f"Machines: {self.manifest['machines']}")
        
    def load_cycle(self, filename):
        """Load a single cycle CSV - FIXED timestamp parsing"""
        df = pd.read_csv(self.data_dir / filename)
        
        # Extract features: torque1, position1, torque2, position2
        features = df[['torque1', 'position1', 'torque2', 'position2']].values
        
        # Fixed: Handle various timestamp formats flexibly
        try:
            # Try ISO8601 first (most common for PostgreSQL timestamps)
            timestamps = pd.to_datetime(df['timestamp'], format='ISO8601')
        except:
            try:
                # Fallback to automatic inference
                timestamps = pd.to_datetime(df['timestamp'])
            except Exception as e:
                print(f"Warning: Could not parse timestamps in {filename}: {e}")
                # Create dummy timestamps if parsing fails
                timestamps = pd.date_range(start='2000-01-01', periods=len(df), freq='500ms')
        
        return features, timestamps
    
    def prepare_data(self, split_strategy='cycle_based'):
        """
        Prepare train/test split
        
        Strategies:
        - 'cycle_based': Split by complete cycles (RECOMMENDED for anomaly detection)
        - 'time_based': First N% cycles for train, last M% for test
        - 'machine_based': Use some machines for train, others for test
        """
        
        print(f"\n{'='*80}")
        print(f"DATA PREPARATION - Strategy: {split_strategy}")
        print(f"{'='*80}")
        
        cycles = self.manifest['cycles']
        self.split_strategy = split_strategy
        
        if split_strategy == 'cycle_based':
            # Randomly split cycles into train/test
            train_cycles, test_cycles = train_test_split(
                cycles, 
                test_size=self.test_size, 
                random_state=42
            )
            
        elif split_strategy == 'time_based':
            # Sort by time and split
            sorted_cycles = sorted(cycles, key=lambda x: x['start_time'])
            split_idx = int(len(sorted_cycles) * (1 - self.test_size))
            train_cycles = sorted_cycles[:split_idx]
            test_cycles = sorted_cycles[split_idx:]
            
        elif split_strategy == 'machine_based':
            # Use machine 1 for train, machine 3 for test, machine 2 for validation
            train_cycles = [c for c in cycles if c['machine_number'] == 1]
            test_cycles = [c for c in cycles if c['machine_number'] == 3]
            
            if len(test_cycles) == 0:
                # Fallback if machine 3 has no cycles
                test_cycles = [c for c in cycles if c['machine_number'] == 2]
        
        print(f"\nTrain/Test Split:")
        print(f"  Training cycles: {len(train_cycles)}")
        print(f"  Testing cycles: {len(test_cycles)}")
        
        if len(test_cycles) == 0:
            raise ValueError("No test cycles found! Check your split strategy or data.")
        
        # Load all training data
        print(f"\nLoading training data...")
        X_train_list = []
        for i, cycle in enumerate(train_cycles):
            try:
                features, _ = self.load_cycle(cycle['filename'])
                X_train_list.append(features)
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{len(train_cycles)} training cycles...")
            except Exception as e:
                print(f"  Warning: Failed to load {cycle['filename']}: {e}")
                continue
        
        if len(X_train_list) == 0:
            raise ValueError("Failed to load any training data!")
        
        # Concatenate all training cycles
        X_train = np.vstack(X_train_list)
        
        # Load all testing data and keep track of cycles
        print(f"\nLoading testing data...")
        X_test_list = []
        self.test_data_by_cycle = []
        
        current_idx = 0
        for i, cycle in enumerate(test_cycles):
            try:
                features, timestamps = self.load_cycle(cycle['filename'])
                X_test_list.append(features)
                
                # Store cycle info for later visualization
                self.test_data_by_cycle.append({
                    'cycle_info': cycle,
                    'features': features,
                    'timestamps': timestamps,
                    'start_idx': current_idx,
                    'length': len(features)
                })
                
                current_idx += len(features)
                
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{len(test_cycles)} test cycles...")
            except Exception as e:
                print(f"  Warning: Failed to load {cycle['filename']}: {e}")
                continue
        
        if len(X_test_list) == 0:
            raise ValueError("Failed to load any test data!")
        
        X_test = np.vstack(X_test_list)
        
        # Scale data
        print(f"\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.X_test_original = X_test  # Keep original scale for visualization
        self.train_cycles = train_cycles
        self.test_cycles = test_cycles
        
        print(f"\nData prepared:")
        print(f"  Training samples: {len(X_train_scaled):,}")
        print(f"  Testing samples: {len(X_test_scaled):,}")
        print(f"  Features: {X_train_scaled.shape[1]}")
        
    def train_models(self, contamination=0.01, use_all_models=False):
        """
        Train anomaly detection models
        
        Parameters:
        - contamination: Expected fraction of anomalies
        - use_all_models: If False, skip slow models for large datasets
        """
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODELS (contamination={contamination})")
        print(f"{'='*80}")
        
        self.contamination = contamination
        n_samples = len(self.X_train)
        
        print(f"Training dataset: {n_samples:,} samples")
        
        # Decision logic based on dataset size
        is_large_dataset = n_samples > 500_000
        
        if is_large_dataset:
            print(f"\n Large dataset detected!")
            if not use_all_models:
                print("  Training only fast models (Isolation Forest, Elliptic Envelope)")
                print("  To train all models, set use_all_models=True (will be slow)")
            else:
                print("  Will subsample data for SVM and LOF")
        
        # 1. Isolation Forest (ALWAYS - scales well)
        print("\n1. Training Isolation Forest...")
        print("   This may take a few minutes...")
        self.models['Isolation Forest'] = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples=min(10000, n_samples),
            n_jobs=-1,
            verbose=0
        )
        self.models['Isolation Forest'].fit(self.X_train)
        print(" Isolation Forest trained")
        
        # 2. Elliptic Envelope (ALWAYS - very fast)
        print("\n2. Training Elliptic Envelope...")
        self.models['Elliptic Envelope'] = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        self.models['Elliptic Envelope'].fit(self.X_train)
        print(" Elliptic Envelope trained")
        
        # 3. One-Class SVM (OPTIONAL - very slow)
        if use_all_models or not is_large_dataset:
            max_svm = min(50000, n_samples)
            print(f"\n3. Training One-Class SVM (on {max_svm:,} samples)...")
            print("   WARNING: This is VERY slow. Press Ctrl+C to skip.")
            
            if n_samples > max_svm:
                print(f"   Subsampling from {n_samples:,} to {max_svm:,} samples...")
                idx = np.random.choice(n_samples, max_svm, replace=False)
                X_svm = self.X_train[idx]
            else:
                X_svm = self.X_train
            
            try:
                self.models['One-Class SVM'] = OneClassSVM(
                    nu=contamination,
                    kernel='rbf',
                    gamma='auto',
                    cache_size=2000
                )
                self.models['One-Class SVM'].fit(X_svm)
                print(" One-Class SVM trained")
            except KeyboardInterrupt:
                print("\n  One-Class SVM training interrupted by user")
        else:
            print("\n3. One-Class SVM: SKIPPED (set use_all_models=True to train)")
        
        # 4. LOF (OPTIONAL - slow)
        if use_all_models or not is_large_dataset:
            max_lof = min(100000, n_samples)
            print(f"\n4. Training Local Outlier Factor (on {max_lof:,} samples)...")
            
            if n_samples > max_lof:
                print(f" Subsampling from {n_samples:,} to {max_lof:,} samples...")
                idx = np.random.choice(n_samples, max_lof, replace=False)
                X_lof = self.X_train[idx]
            else:
                X_lof = self.X_train
            
            self.models['LOF'] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True,
                n_jobs=-1
            )
            self.models['LOF'].fit(X_lof)
            print(" LOF trained")
        else:
            print("\n4. LOF: SKIPPED (set use_all_models=True to train)")
        
        print(f"\n{'='*80}")
        print(f" Training complete! {len(self.models)} models ready.")
        print(f"{'='*80}")
    
    def evaluate_models(self):
        """Evaluate all models on test data"""
        
        print(f"\n{'='*80}")
        print(f"MODEL EVALUATION")
        print(f"{'='*80}")
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            
            # Predict
            predictions = model.predict(self.X_test)
            predictions = (predictions == -1).astype(int)
            
            # Get anomaly scores
            if hasattr(model, 'decision_function'):
                scores = -model.decision_function(self.X_test)
            elif hasattr(model, 'score_samples'):
                scores = -model.score_samples(self.X_test)
            
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # Calculate metrics
            anomaly_count = predictions.sum()
            anomaly_rate = 100 * anomaly_count / len(predictions)
            
            print(f"  Anomalies detected: {anomaly_count:,} / {len(predictions):,}")
            print(f"  Anomaly rate: {anomaly_rate:.2f}%")
            print(f"  Mean anomaly score: {scores.mean():.4f}")
            print(f"  Max anomaly score: {scores.max():.4f}")
            
            # Store results
            self.results[model_name] = {
                'predictions': predictions,
                'scores': scores,
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': f"{anomaly_rate:.2f}%"
            }
    
    def analyze_per_cycle(self):
        """Analyze anomalies per cycle"""
        
        print(f"\n{'='*80}")
        print(f"PER-CYCLE ANOMALY ANALYSIS")
        print(f"{'='*80}")
        
        cycle_analysis = []
        
        for cycle_data in self.test_data_by_cycle:
            cycle = cycle_data['cycle_info']
            cycle_length = cycle_data['length']
            start_idx = cycle_data['start_idx']
            
            # Get predictions for this cycle from each model
            cycle_anomalies = {}
            cycle_scores = {}
            
            for model_name in self.models.keys():
                cycle_preds = self.results[model_name]['predictions'][start_idx:start_idx+cycle_length]
                cycle_score = self.results[model_name]['scores'][start_idx:start_idx+cycle_length]
                cycle_anomalies[f'{model_name}_anomalies'] = int(cycle_preds.sum())
                cycle_scores[f'{model_name}_score'] = float(cycle_score.mean())
            
            cycle_analysis.append({
                'cycle': cycle['filename'],
                'machine': cycle['machine_number'],
                'datapoints': cycle_length,
                'duration_min': cycle['duration_minutes'],
                **cycle_anomalies,
                **cycle_scores
            })
        
        # Create DataFrame
        df_analysis = pd.DataFrame(cycle_analysis)
        
        # Calculate total anomalies across all models
        anomaly_cols = [col for col in df_analysis.columns if col.endswith('_anomalies')]
        df_analysis['total_anomalies'] = df_analysis[anomaly_cols].sum(axis=1)
        df_analysis['avg_anomaly_score'] = df_analysis[[col for col in df_analysis.columns if col.endswith('_score')]].mean(axis=1)
        
        df_analysis = df_analysis.sort_values('total_anomalies', ascending=False)
        
        print("\nTop 10 Most Anomalous Cycles:")
        display_cols = ['cycle', 'machine', 'datapoints', 'total_anomalies', 'avg_anomaly_score']
        print(df_analysis[display_cols].head(10).to_string(index=False))
        
        # Save detailed analysis
        df_analysis.to_csv(self.data_dir / 'cycle_anomaly_analysis.csv', index=False)
        print(f"\n Detailed analysis saved: cycle_anomaly_analysis.csv")
        
        self.cycle_analysis = df_analysis
        return df_analysis

    def analyze_correlations(self, save_plots=True):
        """Analyze feature correlations for normal vs anomalous data"""
        
        print(f"\n{'='*80}")
        print(f"CORRELATION ANALYSIS")
        print(f"{'='*80}")
        
        # Create DataFrame from test data (original scale)
        df = pd.DataFrame(
            self.X_test_original,
            columns=['torque1', 'position1', 'torque2', 'position2']
        )
        
        # Add predictions from first model (typically Isolation Forest)
        model_name = list(self.models.keys())[0]
        df['is_anomaly'] = self.results[model_name]['predictions']
        
        print(f"\nUsing model: {model_name}")
        print(f"Total samples: {len(df):,}")
        print(f"Normal samples: {(df['is_anomaly'] == 0).sum():,}")
        print(f"Anomalous samples: {(df['is_anomaly'] == 1).sum():,}")
        
        # Overall correlation
        print("\n" + "="*60)
        print("OVERALL FEATURE CORRELATIONS")
        print("="*60)
        corr_overall = df[['torque1', 'position1', 'torque2', 'position2']].corr()
        print(corr_overall.round(3))
        
        # Separate normal and anomalous data
        df_normal = df[df['is_anomaly'] == 0][['torque1', 'position1', 'torque2', 'position2']]
        df_anomaly = df[df['is_anomaly'] == 1][['torque1', 'position1', 'torque2', 'position2']]
        
        # Correlations for normal data
        print("\n" + "="*60)
        print(f"NORMAL DATA CORRELATIONS (n={len(df_normal):,})")
        print("="*60)
        corr_normal = df_normal.corr()
        print(corr_normal.round(3))
        
        # Correlations for anomalous data
        print("\n" + "="*60)
        print(f"ANOMALOUS DATA CORRELATIONS (n={len(df_anomaly):,})")
        print("="*60)
        corr_anomaly = df_anomaly.corr()
        print(corr_anomaly.round(3))
        
        # Difference
        print("\n" + "="*60)
        print("CORRELATION DIFFERENCE (Anomaly - Normal)")
        print("="*60)
        corr_diff = corr_anomaly - corr_normal
        print(corr_diff.round(3))
        
        # Identify major differences
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        # Find pairs with largest correlation changes
        diff_pairs = []
        for i in range(len(corr_diff.columns)):
            for j in range(i+1, len(corr_diff.columns)):
                feat1 = corr_diff.columns[i]
                feat2 = corr_diff.columns[j]
                diff = corr_diff.iloc[i, j]
                diff_pairs.append((feat1, feat2, diff))
        
        # Sort by absolute difference
        diff_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("\nTop 3 correlation changes:")
        for feat1, feat2, diff in diff_pairs[:3]:
            normal_corr = corr_normal.loc[feat1, feat2]
            anomaly_corr = corr_anomaly.loc[feat1, feat2]
            print(f"  {feat1} ↔ {feat2}:")
            print(f"    Normal: {normal_corr:+.3f} | Anomaly: {anomaly_corr:+.3f} | Δ: {diff:+.3f}")
        
        if save_plots:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Plot 1: Normal correlations
                sns.heatmap(corr_normal, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, vmin=-1, vmax=1, ax=axes[0], square=True,
                        cbar_kws={'label': 'Correlation'})
                axes[0].set_title(f'Normal Data\n(n={len(df_normal):,})', fontsize=14, pad=15)
                
                # Plot 2: Anomalous correlations
                sns.heatmap(corr_anomaly, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, vmin=-1, vmax=1, ax=axes[1], square=True,
                        cbar_kws={'label': 'Correlation'})
                axes[1].set_title(f'Anomalous Data\n(n={len(df_anomaly):,})', fontsize=14, pad=15)
                
                # Plot 3: Difference
                sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='RdBu_r', 
                        center=0, vmin=-0.5, vmax=0.5, ax=axes[2], square=True,
                        cbar_kws={'label': 'Difference'})
                axes[2].set_title('Correlation Difference\n(Anomaly - Normal)', fontsize=14, pad=15)
                
                plt.suptitle(f'Feature Correlation Analysis - {model_name}', 
                            fontsize=16, y=1.02)
                plt.tight_layout()
                
                # Save plot
                plot_path = self.data_dir / 'correlation_analysis.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"\n Correlation heatmap saved: {plot_path}")
                
                plt.close()
                
            except ImportError:
                print("\n matplotlib/seaborn not available, skipping plot generation")
            except Exception as e:
                print(f"\n Error creating plots: {e}")
        
        # Save correlation data as CSV
        corr_summary = pd.DataFrame({
            'Feature_Pair': [f"{p[0]}-{p[1]}" for p in diff_pairs],
            'Normal_Correlation': [corr_normal.loc[p[0], p[1]] for p in diff_pairs],
            'Anomaly_Correlation': [corr_anomaly.loc[p[0], p[1]] for p in diff_pairs],
            'Difference': [p[2] for p in diff_pairs]
        })
        
        csv_path = self.data_dir / 'correlation_summary.csv'
        corr_summary.to_csv(csv_path, index=False)
        print(f" Correlation summary saved: {csv_path}")
        
        # Store for later use
        self.correlation_analysis = {
            'overall': corr_overall,
            'normal': corr_normal,
            'anomaly': corr_anomaly,
            'difference': corr_diff
        }
        
        return corr_normal, corr_anomaly, corr_diff
    
    def create_interactive_dashboard(self):
        """Create comprehensive interactive HTML dashboard with filters"""
        
        print(f"\n{'='*80}")
        print(f"GENERATING INTERACTIVE VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Create main dashboard HTML
        html_parts = []
        
        # Add CSS and JavaScript for filters
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .controls {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .control-group {
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }
        label {
            font-weight: bold;
            margin-right: 10px;
        }
        select, button {
            padding: 8px 15px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Time-Series Anomaly Detection Dashboard</h1>
        <p>Interactive exploration of anomaly detection results across multiple algorithms</p>
    </div>
""")
        
        # Add statistics cards
        total_test_samples = len(self.X_test)
        total_cycles = len(self.test_cycles)
        
        html_parts.append(f"""
    <div class="stats">
        <div class="stat-card">
            <div class="stat-label">Split Strategy</div>
            <div class="stat-value">{self.split_strategy}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Test Cycles</div>
            <div class="stat-value">{total_cycles}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Test Samples</div>
            <div class="stat-value">{total_test_samples:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Models Trained</div>
            <div class="stat-value">{len(self.models)}</div>
        </div>
    </div>
""")
        
        # Add controls
        model_options = ''.join([f'<option value="{name}">{name}</option>' for name in self.models.keys()])
        cycle_options = ''.join([f'<option value="{i}">Cycle {i+1}: {cycle["cycle_info"]["filename"]}</option>' 
                                for i, cycle in enumerate(self.test_data_by_cycle)])
        
        html_parts.append(f"""
    <div class="controls">
        <h3>Filters & Controls</h3>
        <div class="control-group">
            <label for="modelSelect">Algorithm:</label>
            <select id="modelSelect" onchange="updatePlots()">
                {model_options}
            </select>
        </div>
        <div class="control-group">
            <label for="cycleSelect">Cycle:</label>
            <select id="cycleSelect" onchange="updateCyclePlot()">
                <option value="all">All Cycles (Overview)</option>
                {cycle_options}
            </select>
        </div>
        <div class="control-group">
            <label for="featureSelect">Feature:</label>
            <select id="featureSelect" onchange="updateCyclePlot()">
                <option value="0">Torque 1</option>
                <option value="1">Position 1</option>
                <option value="2">Torque 2</option>
                <option value="3">Position 2</option>
            </select>
        </div>
        <button onclick="downloadData()">📥 Download Results</button>
    </div>
""")
        
        # Add plot containers
        html_parts.append("""
    <div class="plot-container">
        <h3>Model Comparison</h3>
        <div id="comparisonPlot"></div>
    </div>
    
    <div class="plot-container">
        <h3>Anomaly Scores Distribution</h3>
        <div id="scoreDistPlot"></div>
    </div>
    
    <div class="plot-container">
        <h3>Time-Series View</h3>
        <div id="timeSeriesPlot"></div>
    </div>
    
    <div class="plot-container">
        <h3>Per-Cycle Analysis</h3>
        <div id="perCyclePlot"></div>
    </div>
    
    <script>
""")
        
        # Generate JavaScript data
        comparison_data = {
            'models': list(self.models.keys()),
            'anomaly_counts': [self.results[m]['anomaly_count'] for m in self.models.keys()],
            'anomaly_rates': [float(self.results[m]['anomaly_rate'].rstrip('%')) for m in self.models.keys()],
        }
        
        score_data = {
            model: self.results[model]['scores'].tolist()
            for model in self.models.keys()
        }
        
        cycle_data_js = []
        for i, cycle_data in enumerate(self.test_data_by_cycle):
            cycle_length = cycle_data['length']
            start_idx = cycle_data['start_idx']
            
            cycle_dict = {
                'cycle_idx': i,
                'filename': cycle_data['cycle_info']['filename'],
                'machine': cycle_data['cycle_info']['machine_number'],
                'features': cycle_data['features'].tolist(),
                'timestamps': [str(t) for t in cycle_data['timestamps']],
                'predictions': {},
                'scores': {}
            }
            
            for model_name in self.models.keys():
                cycle_dict['predictions'][model_name] = self.results[model_name]['predictions'][start_idx:start_idx+cycle_length].tolist()
                cycle_dict['scores'][model_name] = self.results[model_name]['scores'][start_idx:start_idx+cycle_length].tolist()
            
            cycle_data_js.append(cycle_dict)
        
        per_cycle_summary = self.cycle_analysis.to_dict('records')
        
        html_parts.append(f"""
        // Data from Python
        const comparisonData = {json.dumps(comparison_data)};
        const scoreData = {json.dumps(score_data)};
        const cycleData = {json.dumps(cycle_data_js)};
        const perCycleSummary = {json.dumps(per_cycle_summary)};
        
        // Initialize plots
        function initializePlots() {{
            createComparisonPlot();
            createScoreDistPlot();
            createTimeSeriesPlot();
            createPerCyclePlot();
        }}
        
        // 1. Model Comparison Plot
        function createComparisonPlot() {{
            const trace1 = {{
                x: comparisonData.models,
                y: comparisonData.anomaly_counts,
                type: 'bar',
                name: 'Anomaly Count',
                marker: {{color: '#e74c3c'}}
            }};
            
            const trace2 = {{
                x: comparisonData.models,
                y: comparisonData.anomaly_rates,
                type: 'bar',
                name: 'Anomaly Rate (%)',
                marker: {{color: '#3498db'}},
                yaxis: 'y2'
            }};
            
            const layout = {{
                title: 'Anomaly Detection Comparison Across Models',
                xaxis: {{title: 'Algorithm'}},
                yaxis: {{title: 'Anomaly Count'}},
                yaxis2: {{
                    title: 'Anomaly Rate (%)',
                    overlaying: 'y',
                    side: 'right'
                }},
                barmode: 'group',
                height: 400
            }};
            
            Plotly.newPlot('comparisonPlot', [trace1, trace2], layout);
        }}
        
        // 2. Score Distribution Plot
        function createScoreDistPlot() {{
            const selectedModel = document.getElementById('modelSelect').value;
            
            const trace = {{
                x: scoreData[selectedModel],
                type: 'histogram',
                name: 'Anomaly Scores',
                marker: {{color: '#9b59b6'}},
                nbinsx: 50
            }};
            
            const layout = {{
                title: `Anomaly Score Distribution - ${{selectedModel}}`,
                xaxis: {{title: 'Anomaly Score'}},
                yaxis: {{title: 'Frequency'}},
                height: 400
            }};
            
            Plotly.newPlot('scoreDistPlot', [trace], layout);
        }}
        
        // 3. Time-Series Plot
        function createTimeSeriesPlot() {{
            const selectedModel = document.getElementById('modelSelect').value;
            const selectedCycle = document.getElementById('cycleSelect').value;
            const selectedFeature = parseInt(document.getElementById('featureSelect').value);
            
            const featureNames = ['Torque 1', 'Position 1', 'Torque 2', 'Position 2'];
            
            if (selectedCycle === 'all') {{
                // Show overview of all cycles (sample points for performance)
                let allFeatures = [];
                let allPredictions = [];
                let allScores = [];
                let indices = [];
                
                cycleData.forEach((cycle, idx) => {{
                    const sampleRate = Math.max(1, Math.floor(cycle.features.length / 1000));
                    for (let i = 0; i < cycle.features.length; i += sampleRate) {{
                        allFeatures.push(cycle.features[i][selectedFeature]);
                        allPredictions.push(cycle.predictions[selectedModel][i]);
                        allScores.push(cycle.scores[selectedModel][i]);
                        indices.push(idx * 10000 + i);
                    }}
                }});
                
                const normalIdx = indices.filter((_, i) => allPredictions[i] === 0);
                const anomalyIdx = indices.filter((_, i) => allPredictions[i] === 1);
                const normalVals = normalIdx.map(i => allFeatures[indices.indexOf(i)]);
                const anomalyVals = anomalyIdx.map(i => allFeatures[indices.indexOf(i)]);
                
                const trace1 = {{
                    x: normalIdx,
                    y: normalVals,
                    mode: 'markers',
                    name: 'Normal',
                    marker: {{color: '#3498db', size: 3, opacity: 0.6}}
                }};
                
                const trace2 = {{
                    x: anomalyIdx,
                    y: anomalyVals,
                    mode: 'markers',
                    name: 'Anomaly',
                    marker: {{color: '#e74c3c', size: 6, symbol: 'x'}}
                }};
                
                const layout = {{
                    title: `${{featureNames[selectedFeature]}} - All Cycles Overview (${{selectedModel}})`,
                    xaxis: {{title: 'Sample Index'}},
                    yaxis: {{title: 'Value'}},
                    height: 500,
                    hovermode: 'closest'
                }};
                
                Plotly.newPlot('timeSeriesPlot', [trace1, trace2], layout);
            }} else {{
                // Show specific cycle
                const cycleIdx = parseInt(selectedCycle);
                const cycle = cycleData[cycleIdx];
                
                const features = cycle.features.map(f => f[selectedFeature]);
                const predictions = cycle.predictions[selectedModel];
                const scores = cycle.scores[selectedModel];
                
                const normalIdx = predictions.map((p, i) => p === 0 ? i : -1).filter(i => i >= 0);
                const anomalyIdx = predictions.map((p, i) => p === 1 ? i : -1).filter(i => i >= 0);
                
                const trace1 = {{
                    x: normalIdx,
                    y: normalIdx.map(i => features[i]),
                    mode: 'lines+markers',
                    name: 'Normal',
                    line: {{color: '#3498db', width: 1}},
                    marker: {{size: 4}}
                }};
                
                const trace2 = {{
                    x: anomalyIdx,
                    y: anomalyIdx.map(i => features[i]),
                    mode: 'markers',
                    name: 'Anomaly',
                    marker: {{color: '#e74c3c', size: 8, symbol: 'x'}}
                }};
                
                const trace3 = {{
                    x: Array.from({{length: scores.length}}, (_, i) => i),
                    y: scores,
                    mode: 'lines',
                    name: 'Anomaly Score',
                    yaxis: 'y2',
                    line: {{color: '#f39c12', width: 2}}
                }};
                
                const layout = {{
                    title: `${{cycle.filename}} - ${{featureNames[selectedFeature]}} (${{selectedModel}})`,
                    xaxis: {{title: 'Time Index'}},
                    yaxis: {{title: 'Sensor Value'}},
                    yaxis2: {{
                        title: 'Anomaly Score',
                        overlaying: 'y',
                        side: 'right',
                        range: [0, 1]
                    }},
                    height: 500,
                    hovermode: 'closest'
                }};
                
                Plotly.newPlot('timeSeriesPlot', [trace1, trace2, trace3], layout);
            }}
        }}
        
        // 4. Per-Cycle Analysis Plot
        function createPerCyclePlot() {{
            const selectedModel = document.getElementById('modelSelect').value;
            
            const x = perCycleSummary.map(c => c.cycle);
            const y = perCycleSummary.map(c => c[selectedModel + '_anomalies']);
            const colors = perCycleSummary.map(c => c.machine);
            
            const trace = {{
                x: x,
                y: y,
                type: 'bar',
                marker: {{
                    color: colors,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{title: 'Machine'}}
                }},
                text: y.map(v => v.toString()),
                textposition: 'auto',
            }};
            
            const layout = {{
                title: `Anomalies per Cycle - ${{selectedModel}}`,
                xaxis: {{
                    title: 'Cycle',
                    tickangle: -45,
                    automargin: true
                }},
                yaxis: {{title: 'Number of Anomalies'}},
                height: 500,
                margin: {{b: 150}}
            }};
            
            Plotly.newPlot('perCyclePlot', [trace], layout);
        }}
        
        // Update functions
        function updatePlots() {{
            createScoreDistPlot();
            createTimeSeriesPlot();
            createPerCyclePlot();
        }}
        
        function updateCyclePlot() {{
            createTimeSeriesPlot();
        }}
        
        // Download function
        function downloadData() {{
            const selectedModel = document.getElementById('modelSelect').value;
            const csv = generateCSV(selectedModel);
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `anomaly_results_${{selectedModel.replace(/\\s+/g, '_')}}.csv`;
            a.click();
        }}
        
        function generateCSV(model) {{
            let csv = 'Cycle,Machine,Datapoints,Duration_Min,Anomalies,Avg_Score\\n';
            perCycleSummary.forEach(cycle => {{
                csv += `${{cycle.cycle}},${{cycle.machine}},${{cycle.datapoints}},${{cycle.duration_min}},${{cycle[model + '_anomalies']}},${{cycle[model + '_score']}}\\n`;
            }});
            return csv;
        }}
        
        // Initialize on load
        window.onload = initializePlots;
    </script>
</body>
</html>
""")
        
        # Save HTML file
        html_content = ''.join(html_parts)
        dashboard_path = self.data_dir / 'interactive_dashboard.html'
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n Interactive dashboard created: {dashboard_path}")
        print(f"  Open in browser to explore results with filters!")
        
        return dashboard_path
    
    def save_results(self):
        """Save all results"""
        
        # Save model summary
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Anomalies': result['anomaly_count'],
                'Rate': result['anomaly_rate'],
                'Mean Score': f"{result['scores'].mean():.4f}",
                'Max Score': f"{result['scores'].max():.4f}",
                'Std Score': f"{result['scores'].std():.4f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.data_dir / 'model_summary.csv', index=False)
        print(f"\n Model summary saved: model_summary.csv")
        
        # Save detailed predictions for each model
        for model_name in self.models.keys():
            # Create detailed results DataFrame
            results_df = pd.DataFrame({
                'index': range(len(self.X_test)),
                'torque1': self.X_test_original[:, 0],
                'position1': self.X_test_original[:, 1],
                'torque2': self.X_test_original[:, 2],
                'position2': self.X_test_original[:, 3],
                'prediction': self.results[model_name]['predictions'],
                'anomaly_score': self.results[model_name]['scores']
            })
            
            # Add cycle information
            cycle_labels = []
            for cycle_data in self.test_data_by_cycle:
                cycle_labels.extend([cycle_data['cycle_info']['filename']] * cycle_data['length'])
            results_df['cycle'] = cycle_labels
            
            # Save
            safe_name = model_name.replace(' ', '_').replace('-', '_').lower()
            results_path = self.data_dir / f'predictions_{safe_name}.csv'
            results_df.to_csv(results_path, index=False)
            print(f" Predictions saved: {results_path}")
        
        print(f"\n All results saved in: {self.data_dir}/")
        
    def run_pipeline(self, split_strategy='cycle_based', contamination=0.01, use_all_models=False):
        """Run complete pipeline with interactive visualizations"""
        
        # Step 1: Prepare data
        self.prepare_data(split_strategy=split_strategy)
    
        # Step 2: Train models
        self.train_models(contamination=contamination, use_all_models=use_all_models)
    
        # Step 3: Evaluate models
        self.evaluate_models()
        
        # Step 4: Per-cycle analysis
        cycle_analysis = self.analyze_per_cycle()

        # Step 5: Correlation analysis (NEW!)
        corr_normal, corr_anomaly, corr_diff = self.analyze_correlations(save_plots=True)
        
        # Step 6: Create interactive dashboard
        dashboard_path = self.create_interactive_dashboard()
        
        # Step 7: Save results
        self.save_results()
        
        print(f"\n{'='*80}")
        print(f" PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"\nKey Insights:")
        print(f"  - Trained on {len(self.train_cycles)} cycles")
        print(f"  - Tested on {len(self.test_cycles)} cycles")
        print(f"  - {len(self.models)} models evaluated")
        print(f"  - Results saved in: {self.data_dir}/")
        print(f"\n Interactive Dashboard:")
        print(f"  {dashboard_path}")
        print(f"\nFeatures:")
        print(f"  Filter by Algorithm (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope)")
        print(f"  Filter by Cycle (view individual cycles or overview)")
        print(f"  Filter by Feature (Torque1, Position1, Torque2, Position2)")
        print(f"  Download filtered results as CSV")
        print(f"  Interactive plots with zoom, pan, and hover details")
        
        return self.results, cycle_analysis, dashboard_path


if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists('anomaly_detection_data/dataset_manifest.json'):
        print("ERROR: Dataset not found!")
        print("Please run: python generate_dataset.py first")
        exit(1)
    
    # Initialize pipeline
    pipeline = CycleBasedAnomalyDetection(
        data_dir='anomaly_detection_data',
        test_size=0.2
    )
    
    # FAST MODE: Only train Isolation Forest and Elliptic Envelope
    results, cycle_analysis, dashboard_path = pipeline.run_pipeline(
        split_strategy='machine_based',
        contamination=0.01,
        use_all_models=False  # Set to True to train SVM and LOF (will be VERY slow)
    )
    
    print("\n" + "="*80)
    print("HOW TO SELECT CYCLES FOR TRAINING:")
    print("="*80)
    print("""
    Three strategies are available:
    
    1. 'cycle_based' (RECOMMENDED):
       - Randomly split cycles into 80% train, 20% test
       - Best for general anomaly detection
       - Ensures model sees diverse operational conditions
       - Use when: You want robust detection across all scenarios
    
    2. 'time_based':
       - First 80% of cycles (chronologically) for training
       - Last 20% for testing
       - Good for detecting drift over time
       - Tests on most recent data
       - Use when: You want to detect if newer cycles are anomalous
    
    3. 'machine_based':
       - Train on Machine 1
       - Test on Machine 3
       - Best for testing generalization across machines
       - Useful if machines have different characteristics
       - Use when: You want to deploy trained model to new machines
    
    To change strategy, modify:
    pipeline.run_pipeline(split_strategy='YOUR_CHOICE')
    
    To adjust expected anomaly rate:
    pipeline.run_pipeline(contamination=0.05)  # Expect 5% anomalies
    """)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"""
    1. Open the interactive dashboard:
       {dashboard_path}
    
    2. Explore the visualizations:
       - Compare different algorithms
       - Inspect individual cycles
       - View anomaly score distributions
       - Identify most anomalous cycles
    
    3. Use the filters to:
       - Switch between algorithms
       - Focus on specific cycles
       - Examine different sensors (torque1, position1, etc.)
    
    4. Download results:
       - Click "Download Results" button for CSV export
       - Or use saved files in: anomaly_detection_data/
    
    5. Fine-tune if needed:
       - Adjust contamination parameter
       - Try different split strategies
       - Re-run: python train_anomaly_detection.py
    """)