import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Sklearn fucntions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast 
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Configuration constants
RANDOM_STATE = 42
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS = 30
PRINT_EVERY_N_EPOCHS = 10

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Neural network using Torch. Basic structure via claude, hyperparameters array self adjusted for K-fold cross validated
class SimpleRegressorNet(nn.Module):
    """Simple feedforward neural network for regression"""
    
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(SimpleRegressorNet, self).__init__()
        
        # Architecture:
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Using dropout to try and prevent from overfittign due to small dataset and large feature map
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer for the 1 value: predicted views
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(csv_path, target_var='averageDurationPerUser'):
    """Loads the given csv into a df for basic data structure"""
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Response Var ({target_var}) stats:")
    print(df[target_var].describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    return df

def prepare_features(df):
    """Extract and prepare features for modelling"""
    # Drop rows with missing embeddings (only abt 3 rows)
    df = df.dropna(subset=['text_embedding']).copy()

    # Parsing embeddings from string to array
    print("Parsing embeddings")
    df['embedding_array'] = df['text_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

    # Extract time features from publishDate
    print("Extracting Time Features")
    df['publishDate'] = pd.to_datetime(df['publishDate'])
    df['day_of_week'] = df['publishDate'].dt.dayofweek # 0 for Mon, 6 for Sun
    df['month'] = df['publishDate'].dt.month

    # Fill missing categoricals with 'unknown'
    df['article_type'] = df['article_type'].fillna('unknown')
    df['sneaker_brand'] = df['sneaker_brand'].fillna('unknown')

    # Handle missing numeric values for sneaker price by adding a flag
    df['has_sneaker_price'] = df['sneaker_price'].notna().astype(int)  # 1 if price exists, 0 if not
    df['sneaker_price'] = df['sneaker_price'].fillna(0)  # Fill with 0

    # Combine Jordan Brand, Jordan, and Air Jordan into one category
    df['sneaker_brand'] = df['sneaker_brand'].replace({
        'Jordan Brand': 'Jordan',
        'Air Jordan': 'Jordan'
    })

    # Combine adidas variations (case sensitivity)
    df['sneaker_brand'] = df['sneaker_brand'].replace({
        'adidas': 'Adidas'
    })

    # Create 'Other' category for brands with less than 5 records
    brand_counts = df['sneaker_brand'].value_counts()
    brands_to_keep = brand_counts[brand_counts >= 5].index
    df.loc[~df['sneaker_brand'].isin(brands_to_keep), 'sneaker_brand'] = 'Other'

    print("\nMissing Values Now:")
    print(df.isnull().sum())

    print(f"Total rows after cleaning: {df.shape[0]} rows")
    return df

def create_feature_matrix(df, target_var='averageDurationPerUser', use_log_transform=True):
    """Combine all features into a single matrix for processing"""

    # Get embeddings
    embeddings = np.stack(df['embedding_array'].values)
    print(f"Embeddings shape:{embeddings.shape}")

    # Numeric features
    numeric_cols = ['avg_sentence_length', 'num_sentences', 'num_paragraphs',
        'num_words', 'sneaker_price', 'has_sneaker_price',
        'day_of_week', 'month']

    numeric_features = df[numeric_cols].values
    print(f"Numeric features shape: {numeric_features.shape}")


    # One-hot encode categorical features
    article_type_encoded = pd.get_dummies(df['article_type'], prefix='article')
    brand_encoded = pd.get_dummies(df['sneaker_brand'], prefix='brand')
    categorical_features = pd.concat([article_type_encoded, brand_encoded], axis=1).values
    print(f"Categorical features shape: {categorical_features.shape}")

    X = np.concatenate([embeddings, numeric_features, categorical_features], axis = 1)
    print(f"Total feature shape: {X.shape}")

    # Store category columns for later use in app.py
    category_info = {
        'article_types': article_type_encoded.columns.tolist(),
        'brands': brand_encoded.columns.tolist()
    }

    # Response variable (y-hat)
    y = df[target_var].values

    # Apply log transform to handle skewed distribution
    if use_log_transform:
        y = np.log1p(y)  # log1p(x) = log(1 + x), handles zeros gracefully
        print(f"\nApplied log transform to target variable")
        print(f"Transformed target - Mean: {np.mean(y):.2f}, Std: {np.std(y):.2f}")

    return X, y, category_info



# HELPER FUNCTIONS FOR TRAINING


def prepare_data(X_train, X_test, y_train, y_test):
    """Normalize features and convert to PyTorch tensors"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def train_model_epochs(model, optimizer, criterion, X_train_tensor, y_train_tensor,
                    X_val_tensor=None, y_val_tensor=None, epochs=DEFAULT_EPOCHS, verbose=True):
    """Train model for specified epochs with optional validation"""
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % PRINT_EVERY_N_EPOCHS == 0:
            if X_val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor)
                print(f"Epoch {epoch+1}: Train={loss.item():.2f}, Val={val_loss.item():.2f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.2f}")

    # Return final losses
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train_tensor), y_train_tensor).item()
        val_loss = None
        if X_val_tensor is not None:
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()

    return model, train_loss, val_loss



def evaluate_model(model, X_test_tensor, y_test_tensor, use_log_transform=True):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)

    y_test_np = y_test_tensor.numpy()
    pred_np = predictions.numpy()

    # Log-space metrics
    mse_log = np.mean((y_test_np - pred_np) ** 2)
    rmse_log = np.sqrt(mse_log)
    mae_log = np.mean(np.abs(y_test_np - pred_np))

    ss_res = np.sum((y_test_np - pred_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2_log = 1 - (ss_res / ss_tot)

    metrics = {
        'mse_log': mse_log,
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'r2_log': r2_log
    }

    # Original-scale metrics if using log transform
    if use_log_transform:
        y_test_orig = np.expm1(y_test_np)
        pred_orig = np.expm1(pred_np)

        metrics['mse_orig'] = np.mean((y_test_orig - pred_orig) ** 2)
        metrics['rmse_orig'] = np.sqrt(metrics['mse_orig'])
        metrics['mae_orig'] = np.mean(np.abs(y_test_orig - pred_orig))

        ss_res_orig = np.sum((y_test_orig - pred_orig) ** 2)
        ss_tot_orig = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
        metrics['r2_orig'] = 1 - (ss_res_orig / ss_tot_orig)

    return metrics



# TRAINING FUNCTIONS

def train_model_kfold(X, y, hidden_size=128, dropout_rate=0.3, learning_rate=0.001,
                       weight_decay=DEFAULT_WEIGHT_DECAY, n_splits=5, epochs=50, verbose=True):
    """Train model using K-fold cross validation with specified hyperparameters"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        if verbose:
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Split and prepare data using helper
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        X_train_t, X_val_t, y_train_t, y_val_t, _ = prepare_data(
            X_train_fold, X_val_fold, y_train_fold, y_val_fold
        )

        # Initialize model
        model = SimpleRegressorNet(X.shape[1], hidden_size, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train using helper function
        model, train_loss, val_loss = train_model_epochs(
            model, optimizer, criterion, X_train_t, y_train_t,
            X_val_t, y_val_t, epochs=epochs, verbose=verbose
        )
        fold_scores.append(val_loss)

    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    if verbose:
        print(f"\n=== K-Fold Results ===")
        print(f"Average validation loss: {avg_score:.2f} (+/- {std_score:.2f})")

    return avg_score, std_score


def tune_hyperparameters(X, y, n_splits=5, epochs=50):
    """Test different hyperparameter combinations using k-fold CV"""

    # Define hyperparameters to test - REDUCED for 1k samples with 1.5k features
    hidden_sizes = [32, 64, 128]
    dropout_rates = [0.1, 0.3]
    learning_rates = [0.001, 0.0001]
    weight_decay = DEFAULT_WEIGHT_DECAY  # Regularization

    results = []

    for hidden_size in hidden_sizes:
        for dropout_rate in dropout_rates:
            for lr in learning_rates:
                print(f"\n{'='*60}")
                print(f"Testing: hidden={hidden_size}, dropout={dropout_rate}, lr={lr}")
                print(f"{'='*60}")

                avg_score, std_score = train_model_kfold(
                    X, y,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    n_splits=n_splits,
                    epochs=epochs,
                    verbose=False  # Don't print details for each fold
                )

                print(f"Average Val Loss: {avg_score:.2f} (+/- {std_score:.2f})")

                results.append({
                    'hidden_size': hidden_size,
                    'dropout_rate': dropout_rate,
                    'learning_rate': lr,
                    'avg_val_loss': avg_score,
                    'std_val_loss': std_score
                })

    # Print summary
    print("")
    print("HYPERPARAMETER TUNING RESULTS")
    print("")
    results_df = pd.DataFrame(results).sort_values('avg_val_loss')
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest configuration:")
    print(f"  Hidden size: {int(best['hidden_size'])}")
    print(f"  Dropout rate: {best['dropout_rate']}")
    print(f"  Learning rate: {best['learning_rate']}")
    print(f"  Validation loss: {best['avg_val_loss']:.2f} (+/- {best['std_val_loss']:.2f})")

    return results_df


def train_final_model(X_train, X_test, y_train, y_test, hidden_size, dropout_rate, learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY, epochs=DEFAULT_EPOCHS, save_path=None, category_info=None):
    """Train final model with best hyperparameters and evaluate on hold-out test set"""

    # Prepare data
    X_train_t, X_test_t, y_train_t, y_test_t, scaler = prepare_data(
        X_train, X_test, y_train, y_test
    )

    # Initialize model
    model = SimpleRegressorNet(X_train.shape[1], hidden_size, dropout_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train model
    print("\nTraining final model...")
    model, _, _ = train_model_epochs(
        model, optimizer, criterion, X_train_t, y_train_t,
        epochs=epochs, verbose=True
    )

    # Evaluate model
    metrics = evaluate_model(model, X_test_t, y_test_t, use_log_transform=True)

    # Print results
    print("")
    print("FINAL MODEL EVALUATION")
    print("")
    print(f"\nLog-space:     MAE={metrics['mae_log']:.4f}, R²={metrics['r2_log']:.4f}")
    print(f"Original-scale: MAE={metrics['mae_orig']:,.2f}, R²={metrics['r2_orig']:.4f}")

    # Save model if path provided
    if save_path:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': X_train.shape[1],
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
            'metrics': metrics
        }
        if category_info:
            checkpoint['category_info'] = category_info
        torch.save(checkpoint, save_path)
        print(f"\nModel saved to {save_path}")

    return model, scaler


def load_model(load_path):
    """Load a saved model from disk"""
    # Load with weights_only=False since we're loading scikit-learn objects
    checkpoint = torch.load(load_path, weights_only=False)

    # Reconstruct model
    model = SimpleRegressorNet(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        dropout_rate=checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get scaler, metrics, and category info
    scaler = checkpoint['scaler']
    metrics = checkpoint.get('metrics', None)
    category_info = checkpoint.get('category_info', None)

    print(f"Model loaded from {load_path}")
    if metrics:
        print(f"Saved metrics - MAE: {metrics['mae_log']:.4f}, R²: {metrics['r2_log']:.4f}")

    return model, scaler, category_info



# MAIN EXECUTION
if __name__ == '__main__':
    TARGET_VAR = 'pageViewsTotal'  # Change this to one of the other performance metrics to switch response

    df = load_data("data/hnhh_processed.csv", target_var=TARGET_VAR)
    df = prepare_features(df)

    # CLipping the dataset
    q_hi = df[TARGET_VAR].quantile(0.98)
    df[TARGET_VAR] = np.minimum(df[TARGET_VAR], q_hi)

    X, y, category_info = create_feature_matrix(df, target_var=TARGET_VAR)

    # Split data, creating a hold-out test set
    print("\n")
    print("SPLITTING DATA - Creating hold-out test set")
    print()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set (hold-out): {X_test.shape[0]} samples")

    # Run hyperparameter tuning ONLY on training data
    print("\n")
    print("HYPERPARAMETER TUNING - Using only training data with K-fold CV")
    print()
    results = tune_hyperparameters(X_train, y_train, n_splits=5, epochs=15)

    # Train final model on training data, evaluate on TRUE hold-out test set
    print("\n")
    print("FINAL MODEL TRAINING AND EVALUATION")
    print()
    best = results.iloc[0]
    final_model, final_scaler = train_final_model(
        X_train, X_test, y_train, y_test,
        hidden_size=int(best['hidden_size']),
        dropout_rate=best['dropout_rate'],
        learning_rate=best['learning_rate'],
        weight_decay=DEFAULT_WEIGHT_DECAY,
        epochs=DEFAULT_EPOCHS,
        save_path='scripts/models/pageviews_model.pt',
        category_info=category_info
    )
    
    # Test against a Linear Regression 
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_test = ridge.predict(X_test)
    print("R² log-space (Ridge):", r2_score(y_test, y_pred_test))

    # Transform back to original scale
    y_test_orig = np.expm1(y_test)
    y_pred_test_orig = np.expm1(y_pred_test)
    print("R² original-scale (Ridge):", r2_score(y_test_orig, y_pred_test_orig))