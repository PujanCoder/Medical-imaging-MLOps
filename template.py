from pathlib import Path


PROJECT_NAME = "healthcare-cv-mlops"


DIRECTORIES = [
    ".github/workflows",

    "data/raw/train",
    "data/raw/validation",
    "data/raw/test",
    "data/processed",
    "data/external",

    "notebooks",

    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/monitoring",
    "src/automation",

    "api",

    "tests/unit",
    "tests/integration",

    "configs",

    "models",

    "artifacts/metrics",
    "artifacts/plots",
    "artifacts/reports",

    "monitoring/prometheus",
    "monitoring/grafana",

    "docker",

    "infrastructure/terraform",

    "scripts",
]


FILES = [
    # GitHub Actions
    ".github/workflows/ci.yml",
    ".github/workflows/training.yml",
    ".github/workflows/deployment.yml",

    # Notebooks
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_image_preprocessing.ipynb",
    "notebooks/03_model_experiments.ipynb",
    "notebooks/04_model_evaluation.ipynb",

    # Source code
    "src/__init__.py",

    "src/data/__init__.py",
    "src/data/ingestion.py",
    "src/data/preprocessing.py",

    "src/features/__init__.py",
    "src/features/image_features.py",

    "src/models/__init__.py",
    "src/models/model.py",
    "src/models/train.py",
    "src/models/predict.py",

    "src/evaluation/__init__.py",
    "src/evaluation/evaluate.py",

    "src/monitoring/__init__.py",
    "src/monitoring/data_drift.py",
    "src/monitoring/model_monitor.py",

    "src/automation/__init__.py",
    "src/automation/retraining.py",

    # API
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",

    # Tests
    "tests/unit/test_preprocessing.py",
    "tests/unit/test_model.py",
    "tests/unit/test_api.py",
    "tests/integration/test_pipeline.py",

    # Config
    "configs/config.yaml",
    "configs/model_config.yaml",

    # Monitoring
    "monitoring/prometheus/prometheus.yml",

    # Docker
    "docker/Dockerfile",
    "docker/docker-compose.yml",

    # Infrastructure
    "infrastructure/terraform/.gitkeep",

    # Scripts
    "scripts/download_data.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/run_pipeline.py",

    # Root files
    ".gitignore",
    ".dockerignore",
    ".dvcignore",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
    "README.md",
    "LICENSE",
]


def create_structure():
    root = Path(PROJECT_NAME)

    print(f"\nCreating project: {PROJECT_NAME}\n")

    # Create directories
    for directory in DIRECTORIES:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {path}")

    # Create files
    for file in FILES:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.touch()
            print(f"[FILE] {path}")
        else:
            print(f"[SKIP] {path} already exists")

    # Special files
    create_gitkeep_files(root)

    print("\n----------------------------------------")
    print("Project structure created successfully!")
    print("----------------------------------------")
    print(f"\nLocation: {root.resolve()}")


def create_gitkeep_files(root):
    """
    Create .gitkeep files for directories that would
    otherwise be empty.
    """

    gitkeep_directories = [
        "models",
        "monitoring/grafana",
        "artifacts/metrics",
        "artifacts/plots",
        "artifacts/reports",
    ]

    for directory in gitkeep_directories:
        path = root / directory / ".gitkeep"

        if not path.exists():
            path.touch()
            print(f"[FILE] {path}")


if __name__ == "__main__":
    create_structure()