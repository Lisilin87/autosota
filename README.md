# AutoSOTA

AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI Model Discovery

## Overview

AutoSOTA is an automated research system that takes a paper and its code repository as input, and automatically produces an improved repository with performance better than the original method. The system is organized into three main phases and eight specialized agents, emphasizing resource preparation, experiment reproduction, reflection and innovation, scheduling, and red line constraints.

## Features

- **Automated Paper Reproduction**: Automatically align papers with repositories, dependencies, data, and weights
- **Baseline Execution**: Run baseline experiments and measure metrics
- **Constrained Optimization**: Generate and test optimization ideas under red line constraints
- **Red Line Auditing**: Ensure scientific validity with R1-R7 constraint enforcement
- **State Persistence**: Recover from crashes and resume interrupted runs
- **Docker Isolation**: Run experiments in isolated container environments

## Architecture

The system consists of 8 core agents:

1. **Resource Service**: Aligns papers with repositories, datasets, models, and checkpoints
2. **Objective Service**: Builds tree-structured rubrics for goal setting
3. **Init Service**: Initializes environments and discovers execution commands
4. **Monitor Service**: Monitors execution and detects issues
5. **Fix Service**: Normalizes errors and selects repair strategies
6. **Idea Service**: Generates and manages optimization idea libraries
7. **Scheduler Service**: Manages optimization lifecycle and state persistence
8. **Supervisor Service**: Enforces red line constraints and audits compliance

## Installation

```bash
# Clone the repository
git clone https://github.com/Lisilin87/autosota.git
cd autosota

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Configuration

### Using Qwen API (阿里云通义千问)

The system supports Qwen API from Alibaba Cloud. Configure your `.env` file:

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your Qwen API credentials
QWEN_API_KEY=your_qwen_api_key_here
QWEN_API_BASE=https://coding-intl.dashscope.aliyuncs.com/v1
LLM_PROVIDER=qwen
LLM_MODEL=glm-5
```

### Using OpenAI API

Alternatively, you can use OpenAI API:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
```

## Usage

### Basic Usage

```bash
# Run AutoSOTA on a paper
autosota run \
  --paper-path path/to/paper.pdf \
  --repo-url https://github.com/author/repo \
  --paper-id my-paper-001

# Resume a stopped run
autosota resume --paper-id my-paper-001

# Check status
autosota status --paper-id my-paper-001

# Export results
autosota export --paper-id my-paper-001 --output-dir ./results
```

### Advanced Usage

```bash
# Run with custom configuration
autosota run \
  --paper-path path/to/paper.pdf \
  --repo-url https://github.com/author/repo \
  --config configs/custom.yaml \
  --max-iterations 10 \
  --workspace-root ./workspaces

# Run with paper metadata
autosota run \
  --paper-path path/to/paper.pdf \
  --conference NeurIPS2023 \
  --domain computer-vision \
  --target-metric accuracy
```

## Configuration

Edit `configs/default.yaml` to customize system behavior:

```yaml
workspace_root: ./workspaces
max_iterations: 8

docker:
  enabled: true
  base_image: "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
  gpu_enabled: true

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7

supervisor:
  redline_policy: configs/policies/redlines.yaml
  strict_mode: true
```

## Project Structure

```
autosota/
├── configs/
│   ├── default.yaml
│   ├── prompts/          # Agent prompts
│   └── policies/          # Red line policies
├── src/
│   ├── core/              # Core models and utilities
│   ├── services/          # Service implementations
│   ├── agents/            # Agent implementations
│   ├── runtimes/          # Docker and process runners
│   ├── parsers/           # Paper and repo parsers
│   ├── storage/           # State and artifact storage
│   └── workflows/         # Workflow orchestration
├── workspaces/            # Per-paper workspaces
│   └── {paper_id}/
│       ├── paper/
│       ├── repo/
│       ├── resources/
│       ├── runs/
│       ├── memory/
│       ├── scores.jsonl
│       └── state.json
└── tests/
```

## Output Artifacts

Each AutoSOTA run produces the following artifacts:

- `resource_manifest.json`: Resource preparation results
- `rubric.json`: Objective rubric tree
- `target_metric.json`: Target metric specification
- `init_report.json`: Initialization report
- `code_analysis.md`: Repository analysis
- `idea_library.md`: Optimization idea library
- `research_report.md`: Research report
- `scores.jsonl`: All iteration scores
- `state.json`: System state for recovery
- `best_patch.diff`: Best performing patch
- `final_report.md`: Final summary report

## Red Line Constraints

The system enforces R1-R7 red line constraints:

- **R1**: Evaluation parameters must not be modified
- **R2**: Evaluation scripts must not be modified
- **R3**: Predictions must be from real model inference
- **R4**: Metric balance must be maintained
- **R5**: No test set leakage into training
- **R6**: Dataset distribution must not be modified
- **R7**: Paper-specific constraints must be documented

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_resource_service.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

MIT License - see LICENSE file for details

## Citation

If you use AutoSOTA in your research, please cite:

```bibtex
@article{autosota2024,
  title={AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI Model Discovery},
  author={...},
  journal={...},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the AutoSOTA paper describing an end-to-end automated research system for SOTA AI model discovery.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.
