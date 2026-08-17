"""Configuration management for TrustOps CLI."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_config_path() -> Path:
    """Get the default configuration file path."""
    config_dir = Path.home() / '.trustops'
    config_dir.mkdir(exist_ok=True)
    return config_dir / 'config.json'


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return get_default_config()


def save_config(config: Dict[str, Any], 
                config_path: Optional[str] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Optional path to config file
    """
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'aws': {
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            # No default bucket names: they are globally unique, so a
            # predictable default could be squatted by another account. An
            # empty value here surfaces as a configuration error at use time
            # rather than silently targeting someone else's bucket.
            'datasets_bucket': os.getenv('TRUSTOPS_DATASETS_BUCKET', ''),
            'results_bucket': os.getenv('TRUSTOPS_RESULTS_BUCKET', ''),
            'artifacts_bucket': os.getenv('TRUSTOPS_ARTIFACTS_BUCKET', ''),
            'workflows_table': os.getenv(
                'TRUSTOPS_WORKFLOWS_TABLE', 'trustops-workflows'
            )
        },
        'trust_scoring': {
            'threshold': float(os.getenv('TRUST_SCORE_THRESHOLD', '0.7')),
            'hallucination_similarity_threshold': float(
                os.getenv('HALLUCINATION_SIMILARITY_THRESHOLD', '0.7')
            )
        },
        'models': {
            'default_embedding_model': os.getenv(
                'DEFAULT_EMBEDDING_MODEL', 'amazon.titan-embed-text-v1'
            ),
            'default_foundation_model': os.getenv(
                'DEFAULT_FOUNDATION_MODEL', 'anthropic.claude-v2'
            )
        }
    }


def update_env_from_config(config: Dict[str, Any]) -> None:
    """
    Update environment variables from configuration.
    
    Args:
        config: Configuration dictionary
    """
    aws_config = config.get('aws', {})
    
    if 'region' in aws_config:
        os.environ['AWS_REGION'] = aws_config['region']
    if 'datasets_bucket' in aws_config:
        os.environ['TRUSTOPS_DATASETS_BUCKET'] = aws_config['datasets_bucket']
    if 'results_bucket' in aws_config:
        os.environ['TRUSTOPS_RESULTS_BUCKET'] = aws_config['results_bucket']
    if 'artifacts_bucket' in aws_config:
        os.environ['TRUSTOPS_ARTIFACTS_BUCKET'] = (
            aws_config['artifacts_bucket']
        )
    if 'workflows_table' in aws_config:
        os.environ['TRUSTOPS_WORKFLOWS_TABLE'] = (
            aws_config['workflows_table']
        )
    
    trust_config = config.get('trust_scoring', {})
    if 'threshold' in trust_config:
        os.environ['TRUST_SCORE_THRESHOLD'] = str(trust_config['threshold'])
    if 'hallucination_similarity_threshold' in trust_config:
        os.environ['HALLUCINATION_SIMILARITY_THRESHOLD'] = str(
            trust_config['hallucination_similarity_threshold']
        )
    
    models_config = config.get('models', {})
    if 'default_embedding_model' in models_config:
        os.environ['DEFAULT_EMBEDDING_MODEL'] = (
            models_config['default_embedding_model']
        )
    if 'default_foundation_model' in models_config:
        os.environ['DEFAULT_FOUNDATION_MODEL'] = (
            models_config['default_foundation_model']
        )
